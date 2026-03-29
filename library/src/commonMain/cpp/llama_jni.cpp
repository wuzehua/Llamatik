#include <jni.h>
#include "llama.h"
#include "llama_jni.h"

#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"
#include "common.h"
#include "chat.h"
#include "sampling.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>   // strlen, memcpy
#include <cctype>    // tolower, isalpha, isdigit
#include <cstdlib>   // malloc, free
#include <string_view>
#include <vector>
#include <atomic>
#include <cstdio>
#include <cstdarg>   // va_list, va_start, va_end
#include <filesystem>
#include <system_error>
#include <cerrno>
#include <chrono>
#include <functional>
#include <unistd.h>

// ===================================================================================
//                              PLATFORM LOGGING
// ===================================================================================
//
// Android uses logcat; Desktop uses stderr.
// This file is shared for Android + Desktop builds.

#if defined(__ANDROID__)
#include <android/log.h>
#include <dlfcn.h>
#include <mutex>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "LlamaBridge", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "LlamaBridge", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  "LlamaBridge", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LlamaBridge", __VA_ARGS__)
#else
static void log_stderr(const char* level, const char* fmt, ...) {
    std::fprintf(stderr, "[LlamaBridge][%s] ", level);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}
#define LOGI(...) log_stderr("I", __VA_ARGS__)
#define LOGD(...) log_stderr("D", __VA_ARGS__)
#define LOGW(...) log_stderr("W", __VA_ARGS__)
#define LOGE(...) log_stderr("E", __VA_ARGS__)
#endif

// ===================================================================================
//                              GLOBAL STATE (this TU)
// ===================================================================================

// Embeddings
static struct llama_model *emb_model = nullptr;
static struct llama_context *emb_ctx = nullptr;
static int emb_dim = 0;

// Text generation
static struct llama_model *gen_model = nullptr;
static struct llama_context *gen_ctx = nullptr;
static llama_batch g_step_batch = {};
static bool g_step_batch_inited = false;
static llama_batch g_gen_batch = {};
static bool g_gen_batch_inited = false;
static common_chat_templates_ptr g_gen_chat_templates;
static std::vector<common_chat_msg> g_gen_chat_msgs;
static llama_pos g_gen_system_prompt_position = 0;
static llama_pos g_gen_current_position = 0;
static llama_pos g_gen_stop_generation_position = 0;
static std::string g_gen_cached_token_chars;
static std::ostringstream g_gen_assistant_ss;
static size_t g_gen_context_signature = 0;
static bool g_gen_system_ready = false;
static std::vector<llama_token> g_cached_prompt_tokens;
static std::vector<llama_token> g_cached_prefix_tokens;
static size_t g_cached_session_signature = 0;
static int g_session_cur_pos = 0;
static bool g_session_valid = false;

// Backend lifetime
static bool g_backend_inited = false;

// Streaming cancel flag (for generateStream)
static std::atomic<bool> g_cancel_requested{false};

// Defaults tuned closer to llama.android demo behavior (more stable and faster-to-stop in practice).
static std::atomic<float> g_temperature = 0.30f;
static std::atomic<float> g_top_p = 0.95f;
static std::atomic<int> g_top_k = 40;
static std::atomic<float> g_repeat_penalty = 1.00f;
static std::atomic<int> g_max_new_tokens = 256;

constexpr int N_THREADS_MIN = 2;
constexpr int N_THREADS_MAX = 4;
constexpr int N_THREADS_HEADROOM = 2;
constexpr int DEFAULT_N_BATCH = 512;
constexpr size_t LOG_PREVIEW_MAX = 240;

// ===================================================================================
//                              SMALL HELPERS
// ===================================================================================

static int compute_android_inference_threads() {
    const long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    const int guessed = static_cast<int>(cpu_count > 0 ? cpu_count : N_THREADS_MAX) - N_THREADS_HEADROOM;
    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX, guessed));
    return n_threads;
}

static void ensure_step_batch_ready() {
    if (!g_step_batch_inited) {
        g_step_batch = llama_batch_init(1, 0, 1);
        g_step_batch_inited = true;
    }
}

static void free_step_batch_if_needed() {
    if (g_step_batch_inited) {
        llama_batch_free(g_step_batch);
        g_step_batch_inited = false;
    }
}

static void ensure_gen_batch_ready() {
    if (!g_gen_batch_inited) {
        g_gen_batch = llama_batch_init(DEFAULT_N_BATCH, 0, 1);
        g_gen_batch_inited = true;
    }
}

static void free_gen_batch_if_needed() {
    if (g_gen_batch_inited) {
        llama_batch_free(g_gen_batch);
        g_gen_batch_inited = false;
    }
}

static void reset_generation_prompt_cache() {
    g_cached_prompt_tokens.clear();
    g_cached_prefix_tokens.clear();
    g_cached_session_signature = 0;
    g_session_cur_pos = 0;
    g_session_valid = false;
    g_gen_chat_msgs.clear();
    g_gen_system_prompt_position = 0;
    g_gen_current_position = 0;
    g_gen_stop_generation_position = 0;
    g_gen_cached_token_chars.clear();
    g_gen_assistant_ss.str("");
    g_gen_assistant_ss.clear();
    g_gen_context_signature = 0;
    g_gen_system_ready = false;
}

static void ensure_llama_log_callback() {
    static std::once_flag once;
    std::call_once(once, []() {
        llama_log_set([](ggml_log_level level, const char *text, void *) {
            if (text == nullptr) return;
            switch (level) {
                case GGML_LOG_LEVEL_ERROR:
                    LOGE("%s", text);
                    break;
                case GGML_LOG_LEVEL_WARN:
                    LOGW("%s", text);
                    break;
                case GGML_LOG_LEVEL_INFO:
                    LOGI("%s", text);
                    break;
                default:
                    LOGD("%s", text);
                    break;
            }
        }, nullptr);
        LOGI("llama log callback configured");
    });
}

static void log_file_diagnostics(const char *tag, const char *path) {
    if (path == nullptr || path[0] == '\0') {
        LOGE("%s: empty model path", tag);
        return;
    }

    const int access_result = access(path, R_OK);
    LOGI("%s: model_path=%s readable=%s errno=%d", tag, path, access_result == 0 ? "yes" : "no", errno);

    std::error_code ec;
    const std::filesystem::path fs_path(path);
    const bool exists = std::filesystem::exists(fs_path, ec);
    if (ec) {
        LOGW("%s: exists() failed: %s", tag, ec.message().c_str());
        ec.clear();
    }
    LOGI("%s: exists=%s", tag, exists ? "yes" : "no");
    if (!exists) return;

    const bool is_file = std::filesystem::is_regular_file(fs_path, ec);
    if (ec) {
        LOGW("%s: is_regular_file() failed: %s", tag, ec.message().c_str());
        ec.clear();
    }
    LOGI("%s: regular_file=%s", tag, is_file ? "yes" : "no");

    const auto file_size = std::filesystem::file_size(fs_path, ec);
    if (ec) {
        LOGW("%s: file_size() failed: %s", tag, ec.message().c_str());
        ec.clear();
    } else {
        LOGI("%s: file_size=%llu bytes", tag, static_cast<unsigned long long>(file_size));
    }
}

static void log_registered_backends(const char *tag) {
    const size_t n_backends = ggml_backend_reg_count();
    LOGI("%s: ggml registered backends=%zu", tag, n_backends);
    for (size_t i = 0; i < n_backends; ++i) {
        auto *reg = ggml_backend_reg_get(i);
        const char *name = reg ? ggml_backend_reg_name(reg) : "(null)";
        LOGI("%s: backend[%zu]=%s", tag, i, name ? name : "(null)");
    }
}

static llama_model *load_model_with_fallback(const char *tag, const char *path) {
    llama_model_params params = llama_model_default_params();
    LOGI("%s: trying default load params (mmap=%d mlock=%d check_tensors=%d)",
         tag, params.use_mmap, params.use_mlock, params.check_tensors);
    llama_model *model = llama_model_load_from_file(path, params);
    if (model != nullptr) {
        LOGI("%s: model loaded with default params", tag);
        return model;
    }

    LOGW("%s: default load failed, retrying with mmap disabled", tag);
    params.use_mmap = false;
    params.use_mlock = false;
    model = llama_model_load_from_file(path, params);
    if (model != nullptr) {
        LOGI("%s: model loaded after fallback (mmap=0, mlock=0)", tag);
    } else {
        LOGE("%s: model load failed after fallback", tag);
    }
    return model;
}

static void ensure_android_backends_loaded() {
#if defined(__ANDROID__)
    static std::mutex load_mu;
    std::lock_guard<std::mutex> lock(load_mu);

    if (ggml_backend_reg_count() > 0) {
        return;
    }

    auto try_log_backends = [](const char *stage) -> bool {
        const size_t count = ggml_backend_reg_count();
        LOGI("backend-load stage=%s count=%zu", stage, count);
        for (size_t i = 0; i < count; ++i) {
            auto *reg = ggml_backend_reg_get(i);
            const char *name = reg ? ggml_backend_reg_name(reg) : "(null)";
            LOGI("backend-load stage=%s backend[%zu]=%s", stage, i, name ? name : "(null)");
        }
        return count > 0;
    };

    std::string native_lib_dir;
    bool has_native_lib_dir = false;
    bool looks_like_apk_internal = false;

    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&ensure_android_backends_loaded), &info) != 0 && info.dli_fname != nullptr) {
        const std::string so_path(info.dli_fname);
        LOGI("backend-load: dladdr so path=%s", so_path.c_str());

        const size_t slash = so_path.find_last_of('/');
        if (slash != std::string::npos && slash > 0) {
            native_lib_dir = so_path.substr(0, slash);
            has_native_lib_dir = true;

            // On some Android configurations dladdr can return APK-internal paths like "...base.apk!/lib/arm64-v8a".
            looks_like_apk_internal = native_lib_dir.find('!') != std::string::npos;
            if (!looks_like_apk_internal) {
                LOGI("backend-load: trying ggml_backend_load_all_from_path(%s)", native_lib_dir.c_str());
                ggml_backend_load_all_from_path(native_lib_dir.c_str());
                if (try_log_backends("from_path")) {
                    return;
                }
            } else {
                LOGW("backend-load: skip from_path, APK-internal path is not directly scannable: %s", native_lib_dir.c_str());
            }
        } else {
            LOGW("backend-load: failed to parse native lib dir from %s", so_path.c_str());
        }
    } else {
        LOGW("backend-load: dladdr failed, cannot resolve native lib path");
    }

    LOGW("backend-load: trying ggml_backend_load_all()");
    ggml_backend_load_all();
    if (try_log_backends("load_all")) {
        return;
    }

    // Final fallback: auto-discover ggml backend .so files from native lib directory.
    if (has_native_lib_dir && !looks_like_apk_internal) {
        std::vector<std::string> candidate_paths;
        std::error_code ec;
        for (const auto &entry : std::filesystem::directory_iterator(native_lib_dir, ec)) {
            if (ec) {
                LOGW("backend-load: directory_iterator error on %s: %s",
                     native_lib_dir.c_str(), ec.message().c_str());
                break;
            }
            if (!entry.is_regular_file(ec)) {
                if (ec) ec.clear();
                continue;
            }
            const auto filename = entry.path().filename().string();
            if (filename.rfind("libggml-", 0) == 0 && entry.path().extension() == ".so") {
                candidate_paths.push_back(entry.path().string());
            }
        }

        std::sort(candidate_paths.begin(), candidate_paths.end());
        LOGI("backend-load: discovered %zu ggml backend candidates in %s",
             candidate_paths.size(), native_lib_dir.c_str());
        for (const auto &lib_path : candidate_paths) {
            ggml_backend_reg_t reg = ggml_backend_load(lib_path.c_str());
            LOGI("backend-load: ggml_backend_load(%s) -> %s",
                 lib_path.c_str(), reg ? "ok" : "null");
            if (ggml_backend_reg_count() > 0) {
                break;
            }
        }
    } else {
        LOGW("backend-load: skip filesystem fallback (native_lib_dir unavailable or APK-internal)");
    }

    if (!try_log_backends("filesystem_fallback")) {
        LOGE("backend-load: no ggml backends were loaded after all fallback attempts");
    }
#endif
}

static inline std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::string preview_for_log(const std::string &s, size_t max_len = LOG_PREVIEW_MAX) {
    if (s.empty()) return "";
    std::string out;
    out.reserve(std::min(s.size(), max_len) + 16);
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t') out.push_back(' ');
        else out.push_back(c);
        if (out.size() >= max_len) break;
    }
    if (s.size() > max_len) out += "...";
    return out;
}

static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static int tokenize_with_retry(const llama_vocab *vocab,
        const char *text,
        std::vector<llama_token> &tokens,
        bool add_bos,
        bool parse_special) {
    if (!text) return 0;
    const int text_len = (int) std::strlen(text);

    int n = llama_tokenize(vocab, text, text_len,
            tokens.data(),
            (int) tokens.size(),
            add_bos, parse_special);
    if (n < 0) {
        const int need = -n;
        if (need > 0) {
            tokens.resize(need);
            n = llama_tokenize(vocab, text, text_len,
                    tokens.data(),
                    (int) tokens.size(),
                    add_bos, parse_special);
        }
    }
    return n;
}

static void truncate_to_ctx(std::vector<llama_token> &tokens, int n_ctx, int reserve_tail) {
    if ((int) tokens.size() <= n_ctx - reserve_tail) return;
    const int keep = n_ctx - reserve_tail;
    std::vector<llama_token> out;
    out.reserve(keep);
    out.insert(out.end(), tokens.end() - keep, tokens.end());
    tokens.swap(out);
}

struct PrefillStats {
    size_t prompt_tokens = 0;
    size_t prev_tokens = 0;
    size_t reused_tokens = 0;
    size_t decoded_tokens = 0;
    long long elapsed_ms = 0;
};

// Reuse already-decoded prefix across requests when prompts share a common token prefix.
// This reduces prefill cost dramatically for multi-turn chat where history is repeated each turn.
static bool prefill_prompt_with_cache(
        const std::vector<llama_token> &prompt_tokens,
        int &out_cur_pos,
        PrefillStats *stats = nullptr) {
    if (!gen_ctx || prompt_tokens.empty()) return false;

    const auto t0 = std::chrono::steady_clock::now();
    llama_memory_t mem = llama_get_memory(gen_ctx);

    size_t lcp = 0;
    const size_t n_prev = g_cached_prompt_tokens.size();
    const size_t n_now = prompt_tokens.size();
    while (lcp < n_prev && lcp < n_now && g_cached_prompt_tokens[lcp] == prompt_tokens[lcp]) {
        ++lcp;
    }

    // Full-prefix reuse needs one token re-decode to refresh "last logits" for sampling.
    if (lcp == n_now && n_now > 0) {
        lcp = n_now - 1;
    }

    if (n_prev > 0) {
        if (lcp == 0) {
            llama_memory_clear(mem, false);
        } else if (lcp < n_prev) {
            const bool rm_ok = llama_memory_seq_rm(mem, /*seq_id*/ 0, (llama_pos) lcp, -1);
            if (!rm_ok) {
                LOGW("prefill-cache: seq_rm failed (lcp=%zu prev=%zu), falling back to full clear", lcp, n_prev);
                llama_memory_clear(mem, false);
                lcp = 0;
            }
        }
    }

    const int n_batch_limit = std::max(1, DEFAULT_N_BATCH);
    for (size_t i = lcp; i < n_now; i += (size_t) n_batch_limit) {
        const int cur = (int) std::min((size_t) n_batch_limit, n_now - i);
        llama_batch batch = llama_batch_init(cur, 0, 1);
        batch.n_tokens = cur;

        for (int j = 0; j < cur; ++j) {
            const size_t idx = i + (size_t) j;
            batch.token[j] = prompt_tokens[idx];
            batch.pos[j] = (llama_pos) idx;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (idx + 1 == n_now);
        }

        const int decode_rc = llama_decode(gen_ctx, batch);
        llama_batch_free(batch);
        if (decode_rc != 0) {
            LOGE("prefill-cache: llama_decode failed rc=%d at token_idx=%zu", decode_rc, i);
            llama_memory_clear(mem, false);
            reset_generation_prompt_cache();
            return false;
        }
    }

    g_cached_prompt_tokens = prompt_tokens;
    out_cur_pos = (int) n_now;

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (stats) {
        stats->prompt_tokens = n_now;
        stats->prev_tokens = n_prev;
        stats->reused_tokens = lcp;
        stats->decoded_tokens = n_now - lcp;
        stats->elapsed_ms = ms;
    }
    LOGI("prefill-cache: prompt=%zu prev=%zu reused=%zu decoded=%zu cost=%lld ms",
         n_now, n_prev, lcp, n_now - lcp, (long long) ms);
    return true;
}

// ---------- Sanitizer (strong, used by non-streaming only) ----------
static void drop_lines_with_prefix(std::string &s, const char *prefix_lc) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0, line_start = 0;
    while (i <= s.size()) {
        if (i == s.size() || s[i] == '\n') {
            std::string_view line(s.data() + line_start, i - line_start);
            std::string line_lc = to_lower(std::string(line));
            if (!(line_lc.rfind(prefix_lc, 0) == 0)) {
                out.append(s.data() + line_start, i - line_start);
                if (i != s.size()) out.push_back('\n');
            }
            line_start = i + 1;
        }
        ++i;
    }
    s.swap(out);
}

// returns cleaned answer; fallback if too short or no alpha
static std::string sanitize_generation(std::string s) {
    if (s.empty()) return s;

    for (const char *stop: {"<end_of_turn>", "<|eot_id|>", "</s>"}) {
        size_t p = s.find(stop);
        if (p != std::string::npos) { s = s.substr(0, p); }
    }
    drop_lines_with_prefix(s, "<start_of_turn>");
    drop_lines_with_prefix(s, "<|start_header_id|>");
    drop_lines_with_prefix(s, "<|end_header_id|>");

    {
        std::string sl = to_lower(s);
        size_t qpos = sl.find("question:");
        if (qpos != std::string::npos) s = s.substr(0, qpos);
    }
    {
        std::string sl = to_lower(s);
        size_t cpos = sl.find("context:");
        if (cpos != std::string::npos) s = s.substr(0, cpos);
    }

    auto slice_after_tag = [&](const char *tag) -> bool {
        std::string low = to_lower(s);
        std::string t = to_lower(std::string(tag));
        size_t p = low.find(t);
        if (p != std::string::npos) {
            s = s.substr(p + std::strlen(tag));
            s = trim(s);
            return true;
        }
        return false;
    };
    (void) (slice_after_tag("ANSWER:") || slice_after_tag("FINAL_ANSWER:"));

    s = trim(s);

    auto strip_leading_noise = [](std::string &t) {
        auto ltrim_str = [&](const char *prefix) -> bool {
            size_t n = std::strlen(prefix);
            if (t.size() >= n && std::memcmp(t.data(), prefix, n) == 0) {
                t.erase(0, n);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                return true;
            }
            return false;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            changed |= ltrim_str("• ");
            changed |= ltrim_str("- ");
            changed |= ltrim_str("* ");
            changed |= ltrim_str("> ");
            changed |= ltrim_str(u8"—");
            changed |= ltrim_str(u8"–");

            if (!t.empty() && (t[0] == ':' || t[0] == '-')) {
                t.erase(0, 1);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                changed = true;
            }

            if (t.size() >= 2 && std::isdigit(static_cast<unsigned char>(t[0])) &&
                    (t[1] == '.' || t[1] == ')')) {
                t.erase(0, 2);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                changed = true;
            } else if (t.size() >= 2 && std::isalpha(static_cast<unsigned char>(t[0])) &&
                    (t[1] == '.' || t[1] == ')')) {
                t.erase(0, 2);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                changed = true;
            }
        }

        size_t k = 0;
        while (k < t.size() && !std::isalnum(static_cast<unsigned char>(t[k]))) ++k;
        if (k > 0 && k < t.size()) t.erase(0, k);
    };

    strip_leading_noise(s);
    s = trim(s);

    bool has_alpha = std::any_of(s.begin(), s.end(), [](unsigned char c) {
        return std::isalpha(c);
    });
    if (!has_alpha || s.size() < 12) {
        return "I don't have enough information in my sources.";
    }

    {
        std::string low = to_lower(s);
        const char *fragments[] = {
                "answer only from the provided context",
                "do not repeat the context",
                "respond exactly: \"i don't have enough information in my sources",
                "instructions:",
                "begin your answer",
                "start your response",
                "do not include anything else",
                "reply with only the answer text"
        };
        for (const char *f: fragments) {
            size_t p = low.find(f);
            if (p != std::string::npos) {
                s = trim(s.substr(0, p));
                break;
            }
        }
    }

    return s;
}

static bool build_json_grammar(const char *json_schema, std::string &out_grammar, std::string &out_err) {
    try {
        const std::string schema_str = (json_schema && json_schema[0]) ? std::string(json_schema) : std::string("{}");
        nlohmann::ordered_json schema = nlohmann::ordered_json::parse(schema_str);
        out_grammar = json_schema_to_grammar(schema, /*force_gbnf=*/false);
        return !out_grammar.empty();
    } catch (const std::exception &e) {
        out_err = e.what();
        return false;
    }
}

static std::string build_json_prompt_single(const char *prompt) {
    std::string p = prompt ? prompt : "";
    p += "\n\nReturn ONLY JSON. No markdown, no prose.";
    return p;
}

static std::string build_json_prompt_chat(const std::string &system, const std::string &ctx, const std::string &user, bool has_schema) {
    (void)system; // not used by this simplified prompt builder (kept for signature compatibility)
    std::string p;
    if (!ctx.empty()) {
        p += "Context:\n";
        p += ctx;
        p += "\n\n";
    }
    p += "Request:\n";
    p += user;
    p += "\n\n";
    if (has_schema) {
        p += "Return ONLY JSON matching the provided JSON Schema. No markdown, no prose.";
    } else {
        p += "Return ONLY valid JSON. No markdown, no prose.";
    }
    return p;
}

// ---------- Chat templating ----------
static std::string build_user_with_context(const std::string &context_block,
        const std::string &user_question) {
    auto t = [](const std::string &x) { return trim(x); };
    if (t(context_block).empty()) return "QUESTION:\n" + user_question;
    std::ostringstream oss;
    oss << "CONTEXT:\n" << context_block << "\n\nQUESTION:\n" << user_question;
    return oss.str();
}

static std::string build_chat_prompt_gemma(const std::string &system_msg,
        const std::string &user_msg) {
    std::ostringstream oss;
    const std::string sys = (system_msg.empty()
            ? "You are a careful assistant. Answer ONLY from the provided context. "
              "If the context is insufficient, respond exactly: \"I don't have enough information in my sources.\" "
              "Write 2–5 short sentences in plain text. Do not use bullets or numbering."
            : system_msg + " Write 2–5 short sentences in plain text. Do not use bullets or numbering.");

    oss << "<start_of_turn>system\n"
        << sys
        << "\n<end_of_turn>\n"
        << "<start_of_turn>user\n"
        << user_msg
        << "\n<end_of_turn>\n"
        << "<start_of_turn>model\n"
        << "ANSWER: ";
    return oss.str();
}

static size_t make_session_signature(const std::string &system_msg, const std::string &context_block) {
    std::hash<std::string> h;
    std::string key;
    key.reserve(system_msg.size() + context_block.size() + 1);
    key.append(system_msg);
    key.push_back('\x1f');
    key.append(context_block);
    return h(key);
}

static std::string build_chat_prefix_gemma(const std::string &system_msg, const std::string &context_block) {
    std::ostringstream oss;
    oss << "<start_of_turn>system\n"
        << system_msg
        << "\n<end_of_turn>\n"
        << "<start_of_turn>user\n"
        << "CONTEXT:\n"
        << context_block
        << "\n\nQUESTION:\n";
    return oss.str();
}

static std::string build_chat_user_suffix_gemma(const std::string &user_msg) {
    std::ostringstream oss;
    oss << user_msg
        << "\n<end_of_turn>\n"
        << "<start_of_turn>model\n"
        << "ANSWER: ";
    return oss.str();
}

static bool tokenize_prompt(
        const char *tag,
        const llama_vocab *vocab,
        const std::string &text,
        std::vector<llama_token> &out_tokens,
        bool add_bos,
        bool parse_special) {
    out_tokens.assign(2048, 0);
    int n_tokens = tokenize_with_retry(vocab, text.c_str(), out_tokens, add_bos, parse_special);
    if (n_tokens <= 0) {
        LOGE("%s: tokenize failed", tag);
        return false;
    }
    out_tokens.resize(n_tokens);
    return true;
}

// ===================================================================================
//                                   EMBEDDINGS
// ===================================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_llamatik_library_platform_LlamaBridge_initModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("initModel (embed): %s", path ? path : "(null)");
    ensure_llama_log_callback();
    log_file_diagnostics("initModel(embed)", path);

    ensure_android_backends_loaded();

    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
        log_registered_backends("initModel(embed)");
    }

    emb_model = load_model_with_fallback("initModel(embed)", path);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!emb_model) {
        LOGE("embed model load failed (see llama/ggml logs above for root cause)");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = 2048;
    cparams.n_batch = DEFAULT_N_BATCH;
    cparams.n_ubatch = DEFAULT_N_BATCH;
    cparams.n_threads = compute_android_inference_threads();
    cparams.n_threads_batch = cparams.n_threads;
    LOGI("initModel(embed): n_ctx=%u n_batch=%u n_ubatch=%u n_threads=%d",
         cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, cparams.n_threads);

    emb_ctx = llama_init_from_model(emb_model, cparams);
    if (!emb_ctx) {
        llama_model_free(emb_model);
        emb_model = nullptr;
        return JNI_FALSE;
    }

    emb_dim = llama_model_n_embd(emb_model);
    LOGI("Embed context ready. dim=%d", emb_dim);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_llamatik_library_platform_LlamaBridge_embed(JNIEnv *env, jobject, jstring input) {
    if (!emb_ctx || !emb_model) {
        LOGE("embed: ctx/model null");
        return nullptr;
    }

    const char *inputStr = env->GetStringUTFChars(input, nullptr);
    if (!inputStr) {
        LOGE("embed: input null");
        return nullptr;
    }

    std::vector<llama_token> tokens(1024);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(emb_model),
            inputStr, tokens,
            /*add_bos*/ true,
            /*parse_special*/ false);
    env->ReleaseStringUTFChars(input, inputStr);

    if (n_tokens <= 0 || n_tokens > (int)llama_n_ctx(emb_ctx)) {
        LOGW("embed tokenize fail/too long. n=%d ctx=%u", n_tokens, (unsigned)llama_n_ctx(emb_ctx));
        return nullptr;
    }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }

    if (llama_decode(emb_ctx, batch) != 0) {
        LOGE("embed: llama_decode failed");
        llama_batch_free(batch);
        return nullptr;
    }

    const float *e = llama_get_embeddings_seq(emb_ctx, 0);
    if (!e) {
        LOGE("embed: embeddings null");
        llama_batch_free(batch);
        return nullptr;
    }

    const int dim = llama_model_n_embd(emb_model);
    jfloatArray result = env->NewFloatArray(dim);
    if (!result) {
        llama_batch_free(batch);
        return nullptr;
    }
    env->SetFloatArrayRegion(result, 0, dim, e);
    llama_batch_free(batch);
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_shutdown(JNIEnv *, jobject) {
    if (emb_ctx) llama_free(emb_ctx);
    if (emb_model) llama_model_free(emb_model);
    emb_ctx = nullptr;
    emb_model = nullptr;

    if (gen_ctx) llama_free(gen_ctx);
    if (gen_model) llama_model_free(gen_model);
    gen_ctx = nullptr;
    gen_model = nullptr;
    free_step_batch_if_needed();
    free_gen_batch_if_needed();
    g_gen_chat_templates.reset();
    reset_generation_prompt_cache();

    if (g_backend_inited) {
        llama_backend_free();
        g_backend_inited = false;
    }
}

// ===================================================================================
//                               TEXT GENERATION
// ===================================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_llamatik_library_platform_LlamaBridge_initGenerateModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("initGenerateModel: %s", path ? path : "(null)");
    ensure_llama_log_callback();
    log_file_diagnostics("initGenerateModel", path);

    ensure_android_backends_loaded();

    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
        log_registered_backends("initGenerateModel");
    }

    // Re-init safety: free existing generation resources before loading a new model.
    if (gen_ctx) {
        llama_free(gen_ctx);
        gen_ctx = nullptr;
    }
    if (gen_model) {
        llama_model_free(gen_model);
        gen_model = nullptr;
    }
    free_step_batch_if_needed();
    free_gen_batch_if_needed();
    g_gen_chat_templates.reset();
    reset_generation_prompt_cache();

    gen_model = load_model_with_fallback("initGenerateModel", path);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!gen_model) {
        LOGE("gen model load failed (see llama/ggml logs above for root cause)");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = false;
    cparams.n_ctx = 4096;
    cparams.n_batch = DEFAULT_N_BATCH;
    cparams.n_ubatch = DEFAULT_N_BATCH;
    cparams.n_threads = compute_android_inference_threads();
    cparams.n_threads_batch = cparams.n_threads;
    LOGI("initGenerateModel: n_ctx=%u n_batch=%u n_ubatch=%u n_threads=%d",
         cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, cparams.n_threads);

    gen_ctx = llama_init_from_model(gen_model, cparams);
    if (!gen_ctx) {
        llama_model_free(gen_model);
        gen_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Gen context ready. n_ctx=%u", (unsigned)llama_n_ctx(gen_ctx));
    ensure_step_batch_ready();
    ensure_gen_batch_ready();
    g_gen_chat_templates = common_chat_templates_init(gen_model, "");
    if (!g_gen_chat_templates) {
        LOGW("initGenerateModel: failed to init chat templates, fallback to raw prompts");
    }
    return JNI_TRUE;
}

static const char *default_generation_system_prompt() {
    return "You are a careful assistant. Answer ONLY from the provided context. "
           "If the context is insufficient, respond exactly: \"I don't have enough information in my sources.\" "
           "Write 2–5 short sentences in plain text. Do not use bullets or numbering.";
}

constexpr const char *ROLE_SYSTEM = "system";
constexpr const char *ROLE_USER = "user";
constexpr const char *ROLE_ASSISTANT = "assistant";
constexpr int GEN_OVERFLOW_HEADROOM = 4;

static bool is_valid_utf8(const char *string) {
    if (!string) return true;
    const auto *bytes = (const unsigned char *) string;
    int num;
    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) num = 1;
        else if ((*bytes & 0xE0) == 0xC0) num = 2;
        else if ((*bytes & 0xF0) == 0xE0) num = 3;
        else if ((*bytes & 0xF8) == 0xF0) num = 4;
        else return false;
        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) return false;
            bytes += 1;
        }
    }
    return true;
}

static void reset_generation_short_term_states() {
    g_gen_stop_generation_position = 0;
    g_gen_cached_token_chars.clear();
    g_gen_assistant_ss.str("");
    g_gen_assistant_ss.clear();
}

static void reset_generation_long_term_states(bool clear_kv_cache = true) {
    g_gen_chat_msgs.clear();
    g_gen_system_prompt_position = 0;
    g_gen_current_position = 0;
    g_gen_context_signature = 0;
    g_gen_system_ready = false;
    reset_generation_short_term_states();
    if (clear_kv_cache && gen_ctx) {
        llama_memory_clear(llama_get_memory(gen_ctx), false);
    }
}

static bool shift_generation_context() {
    if (!gen_ctx) return false;
    if (g_gen_current_position <= g_gen_system_prompt_position + 1) return false;
    const int n_discard = (g_gen_current_position - g_gen_system_prompt_position) / 2;
    if (n_discard <= 0) return false;
    LOGI("generate: shifting context, discard=%d", n_discard);
    auto mem = llama_get_memory(gen_ctx);
    llama_memory_seq_rm(mem, 0, g_gen_system_prompt_position, g_gen_system_prompt_position + n_discard);
    llama_memory_seq_add(mem, 0, g_gen_system_prompt_position + n_discard, g_gen_current_position, -n_discard);
    g_gen_current_position -= n_discard;
    g_gen_stop_generation_position = std::max(g_gen_current_position, g_gen_stop_generation_position - n_discard);
    return true;
}

static std::string generation_chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    const bool add_ass = (role == ROLE_USER);
    auto formatted = common_chat_format_single(
            g_gen_chat_templates.get(), g_gen_chat_msgs, new_msg, add_ass, /*use_jinja=*/false);
    g_gen_chat_msgs.push_back(new_msg);
    return formatted;
}

static bool decode_tokens_in_batches_generation(
        const llama_tokens &tokens,
        llama_pos start_pos,
        bool compute_last_logit = false) {
    if (!gen_ctx) return false;
    ensure_gen_batch_ready();
    const int n_ctx = (int) llama_n_ctx(gen_ctx);
    for (int i = 0; i < (int) tokens.size(); i += DEFAULT_N_BATCH) {
        const int cur_batch_size = std::min((int) tokens.size() - i, DEFAULT_N_BATCH);
        common_batch_clear(g_gen_batch);
        if (start_pos + i + cur_batch_size >= n_ctx - GEN_OVERFLOW_HEADROOM) {
            if (!shift_generation_context()) {
                LOGW("generate: context full and shift failed");
                return false;
            }
            start_pos = g_gen_current_position - i;
        }
        for (int j = 0; j < cur_batch_size; ++j) {
            const llama_token tok = tokens[i + j];
            const llama_pos pos = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == (int) tokens.size() - 1);
            common_batch_add(g_gen_batch, tok, pos, {0}, want_logit);
        }
        if (llama_decode(gen_ctx, g_gen_batch) != 0) {
            LOGE("generate: llama_decode failed in batch prefill");
            return false;
        }
    }
    return true;
}

static common_sampler *create_generation_sampler() {
    common_params_sampling sparams;
    sparams.temp = g_temperature.load();
    sparams.top_p = g_top_p.load();
    sparams.top_k = g_top_k.load();
    sparams.penalty_repeat = g_repeat_penalty.load();
    sparams.penalty_last_n = 128;
    LOGI("generate sampler: temp=%.3f top_p=%.3f top_k=%d repeat_penalty=%.3f penalty_last_n=%d",
         sparams.temp, sparams.top_p, sparams.top_k, sparams.penalty_repeat, sparams.penalty_last_n);
    return common_sampler_init(gen_model, sparams);
}

static bool process_system_prompt_internal(const std::string &system_prompt, int &prompt_tokens) {
    if (!gen_ctx || !gen_model) return false;
    reset_generation_long_term_states(true);
    std::string formatted = system_prompt;
    const bool has_chat_template = g_gen_chat_templates && common_chat_templates_was_explicit(g_gen_chat_templates.get());
    if (has_chat_template) {
        formatted = generation_chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }
    LOGI("system prompt: has_template=%s raw_len=%zu formatted_len=%zu raw_preview=\"%s\" formatted_preview=\"%s\"",
         has_chat_template ? "yes" : "no",
         system_prompt.size(),
         formatted.size(),
         preview_for_log(system_prompt).c_str(),
         preview_for_log(formatted).c_str());
    const auto system_tokens = common_tokenize(gen_ctx, formatted, has_chat_template, has_chat_template);
    prompt_tokens = (int) system_tokens.size();
    LOGI("system prompt tokenize: tokens=%d", prompt_tokens);
    if (prompt_tokens <= 0) return false;
    if (!decode_tokens_in_batches_generation(system_tokens, g_gen_current_position, false)) return false;
    g_gen_system_prompt_position = g_gen_current_position = prompt_tokens;
    g_gen_system_ready = true;
    return true;
}

static bool process_user_prompt_internal(const std::string &user_prompt, int predict_len, int &prompt_tokens) {
    if (!gen_ctx || !gen_model) return false;
    reset_generation_short_term_states();
    std::string formatted = user_prompt;
    const bool has_chat_template = g_gen_chat_templates && common_chat_templates_was_explicit(g_gen_chat_templates.get());
    if (has_chat_template) {
        formatted = generation_chat_add_and_format(ROLE_USER, user_prompt);
    }
    LOGI("user prompt: has_template=%s predict_len=%d raw_len=%zu formatted_len=%zu raw_preview=\"%s\" formatted_preview=\"%s\"",
         has_chat_template ? "yes" : "no",
         predict_len,
         user_prompt.size(),
         formatted.size(),
         preview_for_log(user_prompt).c_str(),
         preview_for_log(formatted).c_str());
    auto user_tokens = common_tokenize(gen_ctx, formatted, has_chat_template, has_chat_template);
    prompt_tokens = (int) user_tokens.size();
    LOGI("user prompt tokenize: tokens=%d current_pos_before=%d", prompt_tokens, (int) g_gen_current_position);
    if (prompt_tokens <= 0) return false;
    if (!decode_tokens_in_batches_generation(user_tokens, g_gen_current_position, true)) return false;
    g_gen_current_position += prompt_tokens;
    g_gen_stop_generation_position = g_gen_current_position + predict_len;
    return true;
}

static bool generate_next_token_internal(common_sampler *sampler, std::string &out, std::string &stop_reason) {
    if (!gen_ctx || !gen_model || !sampler) {
        stop_reason = "invalid_state";
        return false;
    }
    const int n_ctx = (int) llama_n_ctx(gen_ctx);
    if (g_gen_current_position >= n_ctx - GEN_OVERFLOW_HEADROOM) {
        if (!shift_generation_context()) {
            stop_reason = "ctx_limit";
            return false;
        }
    }
    if (g_gen_current_position >= g_gen_stop_generation_position) {
        stop_reason = "predict_cap";
        return false;
    }

    const llama_token tok = common_sampler_sample(sampler, gen_ctx, -1);
    common_sampler_accept(sampler, tok, true);

    ensure_gen_batch_ready();
    common_batch_clear(g_gen_batch);
    common_batch_add(g_gen_batch, tok, g_gen_current_position, {0}, true);
    if (llama_decode(gen_ctx, g_gen_batch) != 0) {
        stop_reason = "decode_error";
        return false;
    }
    g_gen_current_position++;

    const auto *vocab = llama_model_get_vocab(gen_model);
    if (llama_vocab_is_eog(vocab, tok)) {
        stop_reason = "eog";
        if (!g_gen_assistant_ss.str().empty()) {
            (void) generation_chat_add_and_format(ROLE_ASSISTANT, g_gen_assistant_ss.str());
        }
        return false;
    }

    const auto piece = common_token_to_piece(gen_ctx, tok);
    g_gen_cached_token_chars += piece;
    if (is_valid_utf8(g_gen_cached_token_chars.c_str())) {
        out += g_gen_cached_token_chars;
        g_gen_assistant_ss << g_gen_cached_token_chars;
        g_gen_cached_token_chars.clear();
    }
    stop_reason = "running";
    return true;
}

static std::string run_generation_from_prompt_tokens(
        const std::vector<llama_token> &prompt_tokens,
        const char *grammar,
        bool sanitize,
        const char *log_tag,
        PrefillStats *prefill_stats = nullptr,
        long long *generation_ms = nullptr) {
    if (!gen_ctx || !gen_model || prompt_tokens.empty()) return "";

    int cur_pos = 0;
    if (!prefill_prompt_with_cache(prompt_tokens, cur_pos, prefill_stats)) return "";

    float temperature = g_temperature.load();
    float top_p = g_top_p.load();
    int top_k = g_top_k.load();
    float repeat_penalty = g_repeat_penalty.load();
    int max_new_tokens = g_max_new_tokens.load();
    const int n_ctx = (int) llama_n_ctx(gen_ctx);
    const llama_vocab *vocab = llama_model_get_vocab(gen_model);
    const bool heuristic_stop_enabled = !(grammar && grammar[0]);

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (grammar && grammar[0]) {
        llama_sampler_chain_add(sampler, llama_sampler_init_grammar(vocab, grammar, "root"));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const auto gen_t0 = std::chrono::steady_clock::now();
    std::string output;
    char buf[8192];
    char sp[64];
    int generated_tokens = 0;
    const char *stop_reason = "max_new_tokens";

    for (int i = 0; i < max_new_tokens; ++i) {
        if (g_cancel_requested.load(std::memory_order_relaxed)) {
            stop_reason = "cancelled";
            break;
        }

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0) {
            stop_reason = "sample_error";
            break;
        }
        if (llama_vocab_is_eog(vocab, tok)) {
            stop_reason = "eog";
            break;
        }
        if (tok == llama_vocab_eos(vocab)) {
            stop_reason = "eos";
            break;
        }

        int sn = llama_token_to_piece(vocab, tok, sp, (int) sizeof(sp), 0, /*special*/ 1);
        if (sn > 0) {
            sp[std::min(sn, (int) sizeof(sp) - 1)] = '\0';
            if (std::strcmp(sp, "<end_of_turn>") == 0 ||
                std::strcmp(sp, "<|eot_id|>") == 0 ||
                std::strcmp(sp, "<start_of_turn>") == 0) {
                stop_reason = "chat_stop_piece";
                break;
            }
        }

        llama_sampler_accept(sampler, tok);

        int nn = llama_token_to_piece(vocab, tok, buf, (int) sizeof(buf), 0, /*special*/ 0);
        if (nn > 0) output.append(buf, nn);

        if (heuristic_stop_enabled && generated_tokens >= 24) {
            const size_t n = output.size();
            const bool ends_with_double_newline = n >= 2 && output[n - 1] == '\n' && output[n - 2] == '\n';
            const bool ends_with_sentence = n >= 1 &&
                    (output[n - 1] == '.' || output[n - 1] == '!' || output[n - 1] == '?');
            if (ends_with_double_newline || (ends_with_sentence && generated_tokens >= 48)) {
                stop_reason = "heuristic_eos";
                break;
            }
        }

        if (cur_pos >= n_ctx) {
            stop_reason = "ctx_limit";
            break;
        }

        ensure_step_batch_ready();
        g_step_batch.n_tokens = 1;
        g_step_batch.token[0] = tok;
        g_step_batch.pos[0] = cur_pos++;
        g_step_batch.n_seq_id[0] = 1;
        g_step_batch.seq_id[0][0] = 0;
        g_step_batch.logits[0] = true;

        if (llama_decode(gen_ctx, g_step_batch) != 0) {
            LOGE("%s: llama_decode failed during generation loop", log_tag);
            stop_reason = "decode_error";
            break;
        }
        g_cached_prompt_tokens.push_back(tok);
        ++generated_tokens;
    }

    const auto gen_t1 = std::chrono::steady_clock::now();
    const long long gen_cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_t1 - gen_t0).count();
    if (generation_ms) *generation_ms = gen_cost_ms;
    LOGI("%s: generation cost=%lld ms generated_tokens=%d max_new_tokens=%d stop_reason=%s",
         log_tag, gen_cost_ms, generated_tokens, max_new_tokens, stop_reason);

    g_session_cur_pos = cur_pos;
    llama_sampler_free(sampler);
    if (sanitize) return sanitize_generation(output);
    return trim(output);
}

static bool ensure_session_prefix(
        const std::string &system,
        const std::string &context,
        const llama_vocab *vocab) {
    const size_t signature = make_session_signature(system, context);
    if (!g_session_valid || g_cached_session_signature != signature || g_cached_prefix_tokens.empty()) {
        reset_generation_prompt_cache();
        std::string prefix = build_chat_prefix_gemma(system, context);
        if (!tokenize_prompt("generateWithContext(prefix)", vocab, prefix, g_cached_prefix_tokens, true, true)) {
            return false;
        }
        const int n_ctx = (int) llama_n_ctx(gen_ctx);
        if ((int) g_cached_prefix_tokens.size() > n_ctx - 8) truncate_to_ctx(g_cached_prefix_tokens, n_ctx, 8);
        PrefillStats prefix_stats{};
        if (!prefill_prompt_with_cache(g_cached_prefix_tokens, g_session_cur_pos, &prefix_stats)) return false;
        g_cached_session_signature = signature;
        g_session_valid = true;
        LOGI("generateWithContext: session initialized signature=%zu prefix_tokens=%zu",
             signature, g_cached_prefix_tokens.size());
        return true;
    }

    PrefillStats rewind_stats{};
    if (!prefill_prompt_with_cache(g_cached_prefix_tokens, g_session_cur_pos, &rewind_stats)) return false;
    LOGI("generateWithContext: session reused signature=%zu prefix_tokens=%zu",
         signature, g_cached_prefix_tokens.size());
    return true;
}

static std::string generate_with_optional_grammar(const char *prompt, const char *grammar, bool sanitize) {
    if (!gen_ctx || !gen_model || !prompt) return "";
    const llama_vocab *vocab = llama_model_get_vocab(gen_model);
    std::vector<llama_token> tokens;
    if (!tokenize_prompt("generate", vocab, prompt, tokens, /*add_bos*/ true, /*parse_special*/ true)) return "";
    const int n_ctx = (int) llama_n_ctx(gen_ctx);
    if ((int) tokens.size() > n_ctx - 8) truncate_to_ctx(tokens, n_ctx, 8);

    PrefillStats stats{};
    long long gen_ms = 0;
    return run_generation_from_prompt_tokens(tokens, grammar, sanitize, "generate", &stats, &gen_ms);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generate(JNIEnv *env, jobject, jstring input) {
    if (!gen_ctx || !gen_model) {
        LOGE("generate: ctx/model null");
        return nullptr;
    }

    const char *prompt = env->GetStringUTFChars(input, nullptr);
    if (!prompt) {
        LOGE("generate: prompt null");
        return nullptr;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string user_prompt = prompt;
    env->ReleaseStringUTFChars(input, prompt);
    LOGI("generate request: len=%zu preview=\"%s\"",
         user_prompt.size(), preview_for_log(user_prompt).c_str());

    reset_generation_long_term_states(true);

    int prompt_tokens = 0;
    const int predict_len = std::max(1, g_max_new_tokens.load());
    const auto prefill_t0 = std::chrono::steady_clock::now();
    if (!process_user_prompt_internal(user_prompt, predict_len, prompt_tokens)) {
        LOGE("generate: process_user_prompt_internal failed");
        return nullptr;
    }
    const auto prefill_t1 = std::chrono::steady_clock::now();
    const long long prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_t1 - prefill_t0).count();

    const auto gen_t0 = std::chrono::steady_clock::now();
    std::string out;
    std::string stop_reason = "predict_cap";
    int generated_tokens = 0;
    common_sampler *sampler = create_generation_sampler();
    while (generate_next_token_internal(sampler, out, stop_reason)) {
        generated_tokens++;
    }
    common_sampler_free(sampler);
    const auto gen_t1 = std::chrono::steady_clock::now();
    const long long generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_t1 - gen_t0).count();

    const auto t1 = std::chrono::steady_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    LOGI("generate perf: prefill=%lld ms prompt_tokens=%d generation=%lld ms generated_tokens=%d stop_reason=%s total=%lld ms",
         prefill_ms,
         prompt_tokens,
         (long long) generation_ms,
         generated_tokens,
         stop_reason.c_str(),
         (long long) total_ms);
    const std::string final_out = trim(out);
    LOGI("generate output: len=%zu preview=\"%s\"",
         final_out.size(), preview_for_log(final_out).c_str());
    return env->NewStringUTF(final_out.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generateWithContext(
        JNIEnv *env, jobject, jstring jSystem, jstring jContext, jstring jUser) {
    if (!gen_ctx || !gen_model) {
        LOGE("generateWithContext: ctx/model null");
        return nullptr;
    }
    if (!jUser) {
        LOGE("generateWithContext: user prompt null");
        return nullptr;
    }

    const char *psys = jSystem ? env->GetStringUTFChars(jSystem, nullptr) : nullptr;
    const char *pctx = jContext ? env->GetStringUTFChars(jContext, nullptr) : nullptr;
    const char *pusr = env->GetStringUTFChars(jUser, nullptr);

    std::string system = psys ? psys : "";
    std::string ctx = pctx ? pctx : "";
    std::string user = pusr ? pusr : "";

    if (jSystem) env->ReleaseStringUTFChars(jSystem, psys);
    if (jContext) env->ReleaseStringUTFChars(jContext, pctx);
    if (jUser) env->ReleaseStringUTFChars(jUser, pusr);

    if (trim(system).empty()) system = default_generation_system_prompt();
    LOGI("generateWithContext request: system_len=%zu context_len=%zu user_len=%zu system_preview=\"%s\" context_preview=\"%s\" user_preview=\"%s\"",
         system.size(),
         ctx.size(),
         user.size(),
         preview_for_log(system).c_str(),
         preview_for_log(ctx).c_str(),
         preview_for_log(user).c_str());

    const auto t0 = std::chrono::steady_clock::now();
    const size_t signature = make_session_signature(system, ctx);
    int system_tokens = 0;
    if (!g_gen_system_ready || g_gen_context_signature != signature) {
        // Put context into system once so each user turn stays compact and deterministic.
        std::string system_with_context = system;
        if (!trim(ctx).empty()) {
            system_with_context += "\n\nUse the following context:\n" + ctx;
        }
        if (!process_system_prompt_internal(system_with_context, system_tokens)) {
            LOGE("generateWithContext: process_system_prompt_internal failed");
            return env->NewStringUTF("");
        }
        g_gen_context_signature = signature;
        LOGI("generateWithContext session: reset signature=%zu", signature);
    } else {
        LOGI("generateWithContext session: reuse signature=%zu", signature);
    }

    int user_tokens = 0;
    const int predict_len = std::max(1, g_max_new_tokens.load());
    const auto prefill_t0 = std::chrono::steady_clock::now();
    if (!process_user_prompt_internal(user, predict_len, user_tokens)) {
        LOGE("generateWithContext: process_user_prompt_internal failed");
        return env->NewStringUTF("");
    }
    const auto prefill_t1 = std::chrono::steady_clock::now();
    const long long prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_t1 - prefill_t0).count();

    const auto gen_t0 = std::chrono::steady_clock::now();
    std::string out;
    std::string stop_reason = "predict_cap";
    int generated_tokens = 0;
    common_sampler *sampler = create_generation_sampler();
    while (generate_next_token_internal(sampler, out, stop_reason)) {
        generated_tokens++;
    }
    common_sampler_free(sampler);
    const auto gen_t1 = std::chrono::steady_clock::now();
    const long long generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_t1 - gen_t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    LOGI("generateWithContext perf: prefill=%lld ms system_tokens=%d user_tokens=%d generation=%lld ms generated_tokens=%d stop_reason=%s total=%lld ms",
         prefill_ms,
         system_tokens,
         user_tokens,
         (long long) generation_ms,
         generated_tokens,
         stop_reason.c_str(),
         (long long) total_ms);
    const std::string final_out = trim(out);
    LOGI("generateWithContext output: len=%zu preview=\"%s\"",
         final_out.size(), preview_for_log(final_out).c_str());
    return env->NewStringUTF(final_out.c_str());
}

// ---------------- JSON constrained (non-streaming) ----------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generateJson(
        JNIEnv *env, jobject, jstring jPrompt, jstring jSchema) {

    const char *pprompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : nullptr;
    const char *pschema = jSchema ? env->GetStringUTFChars(jSchema, nullptr) : nullptr;

    if (!pprompt) {
        if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);
        return nullptr;
    }

    std::string grammar;
    std::string err;
    if (!build_json_grammar(pschema, grammar, err)) {
        env->ReleaseStringUTFChars(jPrompt, pprompt);
        if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);
        return env->NewStringUTF("");
    }

    std::string prompt = build_json_prompt_single(pprompt);

    env->ReleaseStringUTFChars(jPrompt, pprompt);
    if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);

    std::string out = generate_with_optional_grammar(prompt.c_str(), grammar.c_str(), /*sanitize=*/false);
    return env->NewStringUTF(out.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generateJsonWithContext(
        JNIEnv *env, jobject, jstring jSystem, jstring jContext, jstring jUser, jstring jSchema) {

    const char *psys = jSystem ? env->GetStringUTFChars(jSystem, nullptr) : nullptr;
    const char *pctx = jContext ? env->GetStringUTFChars(jContext, nullptr) : nullptr;
    const char *pusr = jUser ? env->GetStringUTFChars(jUser, nullptr) : nullptr;
    const char *pschema = jSchema ? env->GetStringUTFChars(jSchema, nullptr) : nullptr;

    std::string system = psys ? psys : "";
    std::string ctx = pctx ? pctx : "";
    std::string user = pusr ? pusr : "";

    if (jSystem) env->ReleaseStringUTFChars(jSystem, psys);
    if (jContext) env->ReleaseStringUTFChars(jContext, pctx);
    if (jUser) env->ReleaseStringUTFChars(jUser, pusr);

    std::string grammar;
    std::string err;
    if (!build_json_grammar(pschema, grammar, err)) {
        if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);
        return env->NewStringUTF("");
    }
    const bool has_schema = pschema && pschema[0];

    if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);

    std::string prompt = build_json_prompt_chat(system, ctx, user, has_schema);
    std::string out = generate_with_optional_grammar(prompt.c_str(), grammar.c_str(), /*sanitize=*/false);
    return env->NewStringUTF(out.c_str());
}

// ===================================================================================
//                        REAL TOKEN STREAMING (JNI CALLBACKS)
// ===================================================================================

struct StreamMethods {
    jmethodID onDelta;
    jmethodID onComplete;
    jmethodID onError;
};

static bool resolve_stream_methods(JNIEnv *env, jobject cb, StreamMethods &m) {
    jclass cls = env->GetObjectClass(cb);
    if (!cls) return false;
    m.onDelta = env->GetMethodID(cls, "onDelta", "(Ljava/lang/String;)V");
    m.onComplete = env->GetMethodID(cls, "onComplete", "()V");
    m.onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
    return m.onDelta && m.onComplete && m.onError;
}

static inline bool is_eot_piece(const char *s) {
    return std::strcmp(s, "<end_of_turn>") == 0 || std::strcmp(s, "<|eot_id|>") == 0;
}

// Streams tokens from a prepared prompt string.
// Optional: pass a GBNF grammar string to hard-constrain decoding (JSON / JSON schema).
static void stream_from_prompt(
        JNIEnv *env,
        const char *prompt,
        jobject jCallback,
        const StreamMethods &m,
        const char *grammar_gbnf = nullptr) {

    if (!gen_ctx || !gen_model) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("model not initialized"));
        return;
    }

    // Reset cancel flag at the start of each stream
    g_cancel_requested.store(false, std::memory_order_relaxed);

    std::vector<llama_token> tokens(2048);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(gen_model),
            prompt, tokens,
            /*add_bos*/ true,
            /*parse_special*/ true);
    if (n_tokens <= 0) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("tokenization failed"));
        return;
    }
    tokens.resize(n_tokens);

    const int n_ctx = (int)llama_n_ctx(gen_ctx);
    if ((int)tokens.size() > n_ctx - 8) truncate_to_ctx(tokens, n_ctx, 8);
    int cur_pos = 0;
    if (!prefill_prompt_with_cache(tokens, cur_pos)) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("llama_decode failed on prompt"));
        return;
    }

    float temperature    = g_temperature.load();
    float top_p          = g_top_p.load();
    int   top_k          = g_top_k.load();
    float repeat_penalty = g_repeat_penalty.load();
    int   max_new_tokens = g_max_new_tokens.load();

    const llama_vocab *vocab = llama_model_get_vocab(gen_model);

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // IMPORTANT: grammar must be first in the chain so it can veto invalid tokens.
    if (grammar_gbnf && grammar_gbnf[0]) {
        llama_sampler_chain_add(sampler, llama_sampler_init_grammar(vocab, grammar_gbnf, "root"));
    }

    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    char piece_buf[768];
    char spec_buf[64];

    for (int i = 0; i < max_new_tokens; ++i) {
        if (g_cancel_requested.load(std::memory_order_relaxed)) {
            break;
        }

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0) break;
        if (tok == llama_vocab_eos(vocab)) break;

        int sn = llama_token_to_piece(vocab,
                tok, spec_buf, (int) sizeof(spec_buf),
                /* lstrip */ 0, /* special */ 1);
        if (sn > 0) {
            spec_buf[std::min(sn, (int) sizeof(spec_buf) - 1)] = '\0';
            if (is_eot_piece(spec_buf) || std::strcmp(spec_buf, "<start_of_turn>") == 0) {
                break;
            }
        }

        llama_sampler_accept(sampler, tok);

        int nn = llama_token_to_piece(vocab,
                tok, piece_buf, (int) sizeof(piece_buf),
                /* lstrip */ 0, /* special */ 0);
        if (nn > 0) {
            piece_buf[std::min(nn, (int) sizeof(piece_buf) - 1)] = '\0';
            jstring delta = env->NewStringUTF(piece_buf);
            if (delta) {
                env->CallVoidMethod(jCallback, m.onDelta, delta);
                env->DeleteLocalRef(delta);
            }
        }

        if (cur_pos >= n_ctx) break;

        ensure_step_batch_ready();
        g_step_batch.n_tokens = 1;
        g_step_batch.token[0] = tok;
        g_step_batch.pos[0] = cur_pos++;
        g_step_batch.n_seq_id[0] = 1;
        g_step_batch.seq_id[0][0] = 0;
        g_step_batch.logits[0] = true;

        if (llama_decode(gen_ctx, g_step_batch) != 0) {
            env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("llama_decode failed mid-stream"));
            llama_sampler_free(sampler);
            return;
        }
        g_cached_prompt_tokens.push_back(tok);
    }

    llama_sampler_free(sampler);

    // Always signal completion – Kotlin side will ignore if it has nulled activeRequestId
    env->CallVoidMethod(jCallback, m.onComplete);
}

// JNI: stream(prompt, callback)
extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateStream(
        JNIEnv *env, jobject /*thiz*/, jstring jPrompt, jobject jCallback) {

    if (!jPrompt || !jCallback) return;

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateStream: failed to resolve callback methods");
        return;
    }

    const char *prompt = env->GetStringUTFChars(jPrompt, nullptr);
    if (!prompt) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("prompt decode failed"));
        return;
    }

    stream_from_prompt(env, prompt, jCallback, m);
    env->ReleaseStringUTFChars(jPrompt, prompt);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateJsonStream(
        JNIEnv *env, jobject /*thiz*/,
        jstring jPrompt, jstring jSchema,
        jobject jCallback) {

    if (!jPrompt || !jCallback) return;

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateJsonStream: failed to resolve callback methods");
        return;
    }

    const char *prompt = env->GetStringUTFChars(jPrompt, nullptr);
    const char *schema = jSchema ? env->GetStringUTFChars(jSchema, nullptr) : nullptr;

    if (!prompt) {
        if (jSchema) env->ReleaseStringUTFChars(jSchema, schema);
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("prompt decode failed"));
        return;
    }

    std::string grammar;
    std::string err;
    if (!build_json_grammar(schema, grammar, err)) {
        env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jSchema) env->ReleaseStringUTFChars(jSchema, schema);
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF(err.c_str()));
        return;
    }

    std::string wrapped = build_json_prompt_single(prompt);

    env->ReleaseStringUTFChars(jPrompt, prompt);
    if (jSchema) env->ReleaseStringUTFChars(jSchema, schema);

    // FIX: correct argument order + actually use grammar in streaming
    stream_from_prompt(env, wrapped.c_str(), jCallback, m, grammar.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateJsonWithContextStream(
        JNIEnv *env, jobject /*thiz*/,
        jstring jSystem, jstring jContext,
        jstring jUser, jstring jSchema,
        jobject jCallback) {

    if (!jCallback) return;

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateJsonWithContextStream: failed to resolve callback methods");
        return;
    }

    const char *psys = jSystem ? env->GetStringUTFChars(jSystem, nullptr) : nullptr;
    const char *pctx = jContext ? env->GetStringUTFChars(jContext, nullptr) : nullptr;
    const char *pusr = jUser ? env->GetStringUTFChars(jUser, nullptr) : nullptr;
    const char *pschema = jSchema ? env->GetStringUTFChars(jSchema, nullptr) : nullptr;

    std::string system = psys ? psys : "";
    std::string ctx = pctx ? pctx : "";
    std::string user = pusr ? pusr : "";

    if (jSystem) env->ReleaseStringUTFChars(jSystem, psys);
    if (jContext) env->ReleaseStringUTFChars(jContext, pctx);
    if (jUser) env->ReleaseStringUTFChars(jUser, pusr);

    std::string grammar;
    std::string err;
    if (!build_json_grammar(pschema, grammar, err)) {
        if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF(err.c_str()));
        return;
    }
    const bool has_schema = pschema && pschema[0];

    if (jSchema) env->ReleaseStringUTFChars(jSchema, pschema);

    std::string prompt = build_json_prompt_chat(system, ctx, user, has_schema);

    // FIX: correct argument order + actually use grammar in streaming
    stream_from_prompt(env, prompt.c_str(), jCallback, m, grammar.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeCancelGenerate(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    LOGI("nativeCancelGenerate: cancel requested");
    g_cancel_requested.store(true, std::memory_order_relaxed);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateWithContextStream(
        JNIEnv *env, jobject /*thiz*/,
        jstring jSystem, jstring jContext, jstring jUser, jobject jCallback) {

    if (!jCallback) return;

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateWithContextStream: failed to resolve callback methods");
        return;
    }

    const char *psys = jSystem ? env->GetStringUTFChars(jSystem, nullptr) : nullptr;
    const char *pctx = jContext ? env->GetStringUTFChars(jContext, nullptr) : nullptr;
    const char *pusr = jUser ? env->GetStringUTFChars(jUser, nullptr) : nullptr;

    std::string system = psys ? psys : "";
    std::string ctx = pctx ? pctx : "";
    std::string user = pusr ? pusr : "";

    if (jSystem) env->ReleaseStringUTFChars(jSystem, psys);
    if (jContext) env->ReleaseStringUTFChars(jContext, pctx);
    if (jUser) env->ReleaseStringUTFChars(jUser, pusr);

    if (trim(system).empty()) {
        system = "You are a careful assistant. Answer ONLY from the provided context. "
                 "If the context is insufficient, respond exactly: \"I don't have enough information in my sources.\" "
                 "Write 2–5 short sentences in plain text. Do not use bullets or numbering.";
    }

    std::string user_turn = build_user_with_context(ctx, user);
    std::string prompt = build_chat_prompt_gemma(system, user_turn);

    stream_from_prompt(env, prompt.c_str(), jCallback, m);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeUpdateGenerationParams(
        JNIEnv * /*env*/,
        jobject /*thiz*/,
        jfloat temperature,
        jint maxTokens,
        jfloat topP,
        jint topK,
        jfloat repeatPenalty) {

    g_temperature     = temperature;
    g_top_p           = topP;
    g_top_k           = topK;
    g_repeat_penalty  = repeatPenalty;
    g_max_new_tokens  = (int)maxTokens;
}

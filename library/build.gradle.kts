import com.android.build.api.dsl.LibraryExtension
import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import java.io.ByteArrayOutputStream
import javax.inject.Inject

abstract class ConfigureLlamaCmakeTask : DefaultTask() {
    @get:Input
    abstract val cmakePath: Property<String>

    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.ABSOLUTE)
    abstract val sourceDir: DirectoryProperty

    @get:OutputDirectory
    abstract val buildDir: DirectoryProperty

    @get:Input
    abstract val archName: Property<String>

    @get:Input
    abstract val sdkName: Property<String>

    @get:Input
    abstract val minIos: Property<String>

    @get:Input
    abstract val pathEnv: Property<String>

    @get:Inject
    abstract val execOperations: ExecOperations

    @TaskAction
    fun configure() {
        val sdk = when (sdkName.get()) {
            "iPhoneSimulator" -> "iphonesimulator"
            "iPhoneOS" -> "iphoneos"
            else -> "macosx"
        }

        val sdkPathOutput = ByteArrayOutputStream()
        execOperations.exec {
            executable = "xcrun"
            args("--sdk", sdk, "--show-sdk-path")
            standardOutput = sdkPathOutput
        }
        val sdkPath = sdkPathOutput.toString().trim()

        val sourceDirFile = sourceDir.get().asFile
        val buildDirFile = buildDir.get().asFile
        val systemName = if (sdk == "macosx") "Darwin" else "iOS"

        buildDirFile.mkdirs()

        execOperations.exec {
            executable = cmakePath.get()
            environment("PATH", pathEnv.get())
            args(
                "-S", sourceDirFile.absolutePath,
                "-B", buildDirFile.absolutePath,
                "-DCMAKE_SYSTEM_NAME=$systemName",
                "-DCMAKE_OSX_ARCHITECTURES=${archName.get()}",
                "-DCMAKE_OSX_SYSROOT=$sdkPath",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=${minIos.get()}",
                "-DCMAKE_INSTALL_PREFIX=${buildDirFile.resolve("install").absolutePath}",
                "-DCMAKE_IOS_INSTALL_COMBINED=NO",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DGGML_OPENMP=OFF",
                "-DLLAMA_CURL=OFF",
                if (sdk == "iphonesimulator") "-DLLAMA_BUILD_BERT=ON" else "-DLLAMA_BUILD_BERT=OFF",
                if (sdk == "iphonesimulator") "-DLLAMA_BUILD_EMBEDDERS=ON" else "-DLLAMA_BUILD_EMBEDDERS=OFF",
            )
        }
    }
}

abstract class MergeLlamaStaticTask : DefaultTask() {
    @get:Input
    abstract val libtoolPath: Property<String>

    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.ABSOLUTE)
    abstract val libDir: DirectoryProperty

    @get:OutputFile
    abstract val mergedLib: RegularFileProperty

    @get:Inject
    abstract val execOperations: ExecOperations

    @TaskAction
    fun merge() {
        val libRoot = libDir.get().asFile
        val libPath = libRoot.absolutePath

        val whisperCandidates = listOf(
            libRoot.resolve("whisper/src/libwhisper.a"),
            libRoot.resolve("whisper/libwhisper.a"),
            libRoot.resolve("whisper-build/src/libwhisper.a"),
            libRoot.resolve("whisper-build/libwhisper.a")
        )

        val args = mutableListOf(
            "-static",
            "-o", mergedLib.get().asFile.absolutePath,
            "$libPath/libllama_static.a",
            "$libPath/llama-local-build/src/libllama.a",
            "$libPath/llama-local-build/ggml/src/libggml.a",
            "$libPath/llama-local-build/ggml/src/libggml-base.a",
            "$libPath/llama-local-build/ggml/src/libggml-cpu.a",
            "$libPath/llama-local-build/ggml/src/ggml-blas/libggml-blas.a",
            "$libPath/llama-local-build/ggml/src/ggml-metal/libggml-metal.a"
        )

        val whisperLib = whisperCandidates.firstOrNull { it.exists() }
        if (whisperLib != null) {
            args += whisperLib.absolutePath
        } else {
            logger.warn("Whisper static library not found in $libPath. iOS voice/STT symbols will NOT be linked.")
        }

        execOperations.exec {
            executable = libtoolPath.get()
            args(args)
        }
    }
}

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.compose.compiler)
    id("org.jetbrains.compose")
    id("com.android.library")
    id("org.jetbrains.dokka") version "2.1.0"
    id("maven-publish")
    id("signing")
}

group = "com.llamatik"
version = (System.getenv("RELEASE_VERSION") ?: "0.0.0-SNAPSHOT")

// Choose ONE min iOS version and use it everywhere
val minIosVersion = "16.6"

kotlin {
    // ---- ANDROID target MUST publish a library variant (AAR) ----
    androidTarget {
        // This is the key bit that ensures the Android actuals (and AAR) are published
        publishLibraryVariants("release")
    }

    // JVM target (if you want JVM consumer artifacts)
    jvm()

    // iOS targets
    iosX64()
    iosArm64()
    iosSimulatorArm64()

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget> {
        binaries.framework {
            baseName = "llamatik"
            isStatic = true
            linkerOpts("-Wl,-no_implicit_dylibs")
            freeCompilerArgs += listOf("-Xbinary=bundleId=com.llamatik.library")
            freeCompilerArgs += "-Xoverride-konan-properties=osVersionMin.ios=16.6"
        }
    }

    fun findTool(name: String, extraCandidates: List<String> = emptyList()): String {
        System.getenv("${name.uppercase()}_PATH")?.let { if (file(it).canExecute()) return it }
        val candidates = mutableListOf(
            "/opt/homebrew/bin/$name",   // Apple Silicon Homebrew
            "/usr/local/bin/$name",      // Intel Homebrew or manual install
            "/usr/bin/$name"             // system (libtool lives here)
        )
        candidates.addAll(extraCandidates)
        try {
            val out = providers.exec { commandLine("which", name) }
                .standardOutput.asText.get().trim()
            if (out.isNotEmpty() && file(out).canExecute()) return out
        } catch (_: Throwable) {}
        for (p in candidates) if (file(p).canExecute()) return p
        throw GradleException(
            "Cannot find required tool '$name'. " +
                    "Install it (e.g. 'brew install $name') or set ${name.uppercase()}_PATH=/full/path/to/$name"
        )
    }

    // Resolve tools once
    val cmakeExecutablePath = findTool("cmake")
    val libtoolExecutablePath = findTool("libtool") // should be /usr/bin/libtool on macOS
    val toolPathEnv = "/opt/homebrew/bin:${System.getenv("PATH") ?: ""}"

    listOf(
        Triple(iosX64(), "x86_64", "iPhoneSimulator"),
        Triple(iosArm64(), "arm64", "iPhoneOS"),
        Triple(iosSimulatorArm64(), "arm64", "iPhoneSimulator")
    ).forEach { (arch, archValue, sdkValue) ->
        val cmakeBuildDirProvider = layout.buildDirectory.dir("llama-cmake/$sdkValue/${arch.name}")
        val cmakeBuildDir = cmakeBuildDirProvider.get().asFile
        val buildTaskName = "buildLlamaCMake${arch.name.replaceFirstChar { it.uppercase() }}"

        val buildTask = tasks.register(buildTaskName, ConfigureLlamaCmakeTask::class) {
            cmakePath.set(cmakeExecutablePath)
            sourceDir.set(layout.projectDirectory.dir("cmake/llama-wrapper"))
            buildDir.set(cmakeBuildDirProvider)
            archName.set(archValue)
            sdkName.set(sdkValue)
            minIos.set(minIosVersion)
            pathEnv.set(toolPathEnv)
        }

        val compileTask = tasks.register(
            "compileLlamaCMake${arch.name.replaceFirstChar { it.uppercase() }}",
            Exec::class
        ) {
            dependsOn(buildTask)
            environment("PATH", toolPathEnv)
            executable = cmakeExecutablePath
            args(
                "--build", cmakeBuildDir.absolutePath,
                "--target", "llama_static_wrapper",
                "--verbose"
            )
        }

        val libPath = cmakeBuildDir.absolutePath

        val mergeTask = tasks.register(
            "mergeLlamaStatic${arch.name.replaceFirstChar { it.uppercase() }}",
            MergeLlamaStaticTask::class
        ) {
            dependsOn(compileTask)
            libtoolPath.set(libtoolExecutablePath)
            libDir.set(cmakeBuildDirProvider)
            mergedLib.set(layout.buildDirectory.file("llama-cmake/$sdkValue/${arch.name}/libllama_merged.a"))
        }

        // Ensure cinterop runs after the native libs are built/merged
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.CInteropProcess>().configureEach {
            dependsOn(mergeTask)
        }

        arch.compilations.getByName("main").cinterops {
            create("llama") {
                val defFileName = "llama_ios.def"

                defFile("src/iosMain/c_interop/$defFileName")
                packageName("com.llamatik.library.platform.llama")

                compilerOpts("-I${projectDir}/src/iosMain/c_interop/include")

                extraOpts(
                    "-libraryPath", libPath
                )

                tasks.named(interopProcessingTaskName).configure {
                    dependsOn(mergeTask)
                }
            }

            create("whisper") {
                val defFileName = "whisper_ios.def"

                defFile("src/iosMain/c_interop/$defFileName")
                packageName("com.llamatik.library.platform.whisper")

                compilerOpts("-I${projectDir}/src/iosMain/c_interop/include")

                extraOpts(
                    "-libraryPath", libPath
                )

                tasks.named(interopProcessingTaskName).configure {
                    dependsOn(mergeTask)
                }
            }
        }

        val merged = "$libPath/libllama_merged.a"

        arch.binaries.getFramework("DEBUG").apply {
            baseName = "llamatik"
            isStatic = true
            linkerOpts(
                "-L$libPath",
                "-Wl,-force_load", merged,
                "-framework", "Accelerate",
                "-framework", "Metal",
                "-Wl,-no_implicit_dylibs",
                if (sdkValue.contains("Simulator"))
                    "-mios-simulator-version-min=$minIosVersion"
                else
                    "-mios-version-min=$minIosVersion"
            )
        }
        arch.binaries.getFramework("RELEASE").apply {
            baseName = "llamatik"
            isStatic = true
            linkerOpts(
                "-L$libPath",
                "-Wl,-force_load", merged,
                "-framework", "Accelerate",
                "-framework", "Metal",
                "-Wl,-no_implicit_dylibs",
                if (sdkValue.contains("Simulator"))
                    "-mios-simulator-version-min=$minIosVersion"
                else
                    "-mios-version-min=$minIosVersion"
            )
        }
    }

    // ---------- Desktop (JVM) JNI build for llama_jni (macOS/Linux/Windows) ----------

    // Detect host OS (for build output folder naming only)
    val hostOsName = System.getProperty("os.name").lowercase()
    val desktopPlatform = when {
        hostOsName.contains("mac") -> "macos"
        hostOsName.contains("linux") -> "linux"
        hostOsName.contains("win") -> "windows"
        else -> error("Unsupported desktop OS: $hostOsName")
    }

    // Output: library/build/llama-jni/<platform>/{libllama_jni.dylib|so|dll}
    val desktopJniBuildDir = layout.buildDirectory
        .dir("llama-jni/$desktopPlatform")
        .get()
        .asFile

    val desktopJniSourceDir = projectDir.resolve("cmake/llama-jni-desktop")

    val buildLlamaJniDesktop by tasks.registering(Exec::class) {
        group = "llama-native"
        description = "Configure CMake for desktop ($desktopPlatform) llama_jni"

        doFirst {
            if (!desktopJniSourceDir.resolve("CMakeLists.txt").exists()) {
                throw GradleException(
                    "Desktop JNI CMakeLists.txt not found at: ${desktopJniSourceDir.resolve("CMakeLists.txt").absolutePath}\n" +
                            "Expected a CMake project under library/src/commonMain/cpp"
                )
            }

            desktopJniBuildDir.mkdirs()

            val args = mutableListOf(
                cmakeExecutablePath,
                "-S", desktopJniSourceDir.absolutePath,
                "-B", desktopJniBuildDir.absolutePath,
                "-DCMAKE_BUILD_TYPE=Release"
            )

            // Optional: help CMake on macOS when run from CI
            if (desktopPlatform == "macos") {
                args += listOf("-DCMAKE_SYSTEM_NAME=Darwin")
            }

            commandLine(args)
        }
    }

    val compileLlamaJniDesktop by tasks.registering(Exec::class) {
        group = "llama-native"
        description = "Build desktop ($desktopPlatform) llama_jni native library"
        dependsOn(buildLlamaJniDesktop)

        commandLine(
            cmakeExecutablePath,
            "--build", desktopJniBuildDir.absolutePath,
            "--config", "Release"
        )
    }

    val libFileName = System.mapLibraryName("llama_jni") // mac: libllama_jni.dylib, linux: libllama_jni.so, win: llama_jni.dll

    val generatedNativeResourcesDir = layout.buildDirectory.dir("generated/native-resources").get().asFile

    val copyDesktopJniToResources by tasks.registering(Copy::class) {
        group = "llama-native"
        dependsOn(compileLlamaJniDesktop)

        val outDir = generatedNativeResourcesDir.resolve("native/$desktopPlatform")
        from(desktopJniBuildDir.resolve(libFileName))
        into(outDir)

        outputs.dir(outDir) // ✅ helps Gradle validation

        doFirst {
            val f = desktopJniBuildDir.resolve(libFileName)
            if (!f.exists()) throw GradleException("Desktop JNI output not found: ${f.absolutePath}")
        }
    }

    tasks.matching { it.name == "jvmProcessResources" }.configureEach {
        dependsOn(copyDesktopJniToResources)
    }

    tasks.matching { it.name == "compileKotlinJvm" }.configureEach {
        dependsOn(compileLlamaJniDesktop)
        dependsOn(copyDesktopJniToResources)
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(libs.kotlin.stdlib)
                implementation(compose.ui)
                implementation(compose.foundation)
                implementation(compose.components.resources)
                resources.srcDir("src/commonMain/resources")
                resources.exclude("**/*.gguf")
            }
        }
        val commonTest by getting {
            dependencies { implementation(libs.kotlin.test) }
        }
        val androidMain by getting
        val jvmMain by getting {
            resources.srcDir(generatedNativeResourcesDir)
        }
    }
}

compose.resources {
    publicResClass = true
}

extensions.configure<LibraryExtension> {
    namespace = "com.llamatik"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
        ndk {
            abiFilters += setOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DLLAMA_BUILD_COMMON=ON",
                    "-DLLAMA_OPENSSL=OFF",
                    "-DGGML_NATIVE=OFF",
                    "-DGGML_BACKEND_DL=ON",
                    "-DGGML_CPU_ALL_VARIANTS=ON",
                    "-DGGML_LLAMAFILE=OFF"
                )
            }
        }
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release { isMinifyEnabled = false }
        debug { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    packaging {
        jniLibs { useLegacyPackaging = false }
    }

    sourceSets.getByName("main").jniLibs.srcDirs("src/commonMain/jniLibs")

    externalNativeBuild {
        cmake {
            path = file("src/commonMain/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
            withJavadocJar()
        }
    }
}

val dokkaPubHtmlTask: Task? = tasks.findByName("dokkaGeneratePublicationHtml")
val dokkaAllHtmlTask: Task? = tasks.findByName("dokkaGenerateHtml")

val dokkaHtmlDir = if (dokkaPubHtmlTask != null) {
    layout.buildDirectory.dir("dokka/htmlPublication")
} else {
    layout.buildDirectory.dir("dokka/html")
}

val javadocJar by tasks.registering(Jar::class) {
    group = JavaBasePlugin.DOCUMENTATION_GROUP
    archiveClassifier.set("javadoc")
    dokkaPubHtmlTask?.let { dependsOn(it) } ?: dokkaAllHtmlTask?.let { dependsOn(it) }

    from(dokkaHtmlDir)
}

publishing {
    publications.withType<MavenPublication>().configureEach {
        pom {
            name.set("Llamatik")
            description.set("Kotlin Multiplatform library for LLaMA/LLM inference.")
            url.set("https://github.com/ferranpons/llamatik")
            licenses {
                license {
                    name.set("Apache-2.0")
                    url.set("https://www.apache.org/licenses/LICENSE-2.0")
                }
            }
            developers {
                developer {
                    id.set("ferranpons")
                    name.set("Ferran Pons")
                    url.set("https://github.com/ferranpons")
                }
            }
            scm {
                url.set("https://github.com/ferranpons/llamatik")
                connection.set("scm:git:git://github.com/ferranpons/llamatik.git")
                developerConnection.set("scm:git:ssh://github.com/ferranpons/llamatik.git")
            }
        }

        if (name.contains("jvm", ignoreCase = true)) {
            artifact(javadocJar)
        }
    }
}

signing {
    useInMemoryPgpKeys(
        (findProperty("signingInMemoryKey") as String?) ?: System.getenv("SIGNING_KEY"),
        (findProperty("signingInMemoryKeyPassword") as String?) ?: System.getenv("SIGNING_PASSWORD")
    )
    sign(publishing.publications)
}

afterEvaluate {
    tasks.named("publish") {
        dependsOn(
            "linkDebugFrameworkIosArm64",
            "linkDebugFrameworkIosX64",
            "linkDebugFrameworkIosSimulatorArm64"
        )
        dependsOn("assembleRelease")
    }
}

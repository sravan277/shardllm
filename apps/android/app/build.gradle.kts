plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.dllm.mesh"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.dllm.mesh"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-phase0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "2.0.20"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    // EXTRA vs spec (required to compile viewModel() in composables):
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    // EXTRA vs spec (required for NavHost two-route navigation):
    implementation("androidx.navigation:navigation-compose:2.7.7")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:okhttp-sse:4.12.0")
    implementation("androidx.datastore:datastore-preferences:1.1.1")
    implementation("androidx.camera:camera-core:1.5.3")
    implementation("androidx.camera:camera-camera2:1.5.3")
    implementation("androidx.camera:camera-lifecycle:1.5.3")
    implementation("androidx.camera:camera-view:1.5.3")
    // EXTRA vs spec (MlKitAnalyzer lives here, not in camera-view):
    implementation("androidx.camera:camera-mlkit-vision:1.5.3")
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
    implementation("androidx.work:work-runtime-ktx:2.9.0")

    debugImplementation("androidx.compose.ui:ui-tooling")
}

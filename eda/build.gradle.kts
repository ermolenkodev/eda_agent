group = rootProject.group
version = rootProject.version

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public")
}

plugins {
    kotlin("jvm") version "2.1.10"
    alias(libs.plugins.kotlin.serialization)
}

dependencies {
    implementation(libs.grazie.client)
    implementation(libs.koog.agents)
    implementation(libs.logback.classic)
}

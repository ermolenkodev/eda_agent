rootProject.name = "eda_agent"

pluginManagement {
    resolutionStrategy {
    }

    repositories {
        gradlePluginPortal()
        maven(url = "https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public")
    }
}

include(":eda")

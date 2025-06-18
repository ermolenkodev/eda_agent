package eda

import ai.jetbrains.code.prompt.executor.clients.grazie.koog.model.JetBrainsAIModels
import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.tools.Tool
import ai.koog.agents.core.tools.ToolArgs
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.core.tools.reflect.asTools
import ai.koog.agents.features.eventHandler.feature.handleEvents
import ai.koog.book.utils.simpleGrazieExecutor
import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.system.exitProcess

private const val PYTHON_KERNEL_PATH = "eda/src/main/kernel"

fun main() = runBlocking {
    // --- Kernel Process Setup ---
    if (!File(PYTHON_KERNEL_PATH).exists()) {
        println("FATAL ERROR: Python kernel script not found at '$PYTHON_KERNEL_PATH'.")
        println("Please save the `kernel.py` file and update the PYTHON_KERNEL_PATH constant in `EdaAgent.kt`.")
        exitProcess(1)
    }

    val process = ProcessBuilder("python", "-u", PYTHON_KERNEL_PATH)
        .start()

    val kernelInput = process.outputStream
    val kernelOutput = process.inputStream.bufferedReader()

    // --- Agent Setup ---
    val agentContext = AgentContext()
    val toolset = EdaTools(agentContext, kernelInput, kernelOutput)
    val toolRegistry = ToolRegistry {
        tools(toolset.asTools())
    }

    val apiKey = System.getenv("GRAZIE_TOKEN") ?: run {
        println("Error: GRAZIE_TOKEN environment variable not set.")
        process.destroy()
        return@runBlocking
    }

    val agent = AIAgent(
        executor = simpleGrazieExecutor(apiKey),
        llmModel = JetBrainsAIModels.OpenAI.GPT4o,
        systemPrompt = "You are a helpful data analysis assistant.", // This will be replaced dynamically.
        toolRegistry = toolRegistry,
        strategy = edaStrategy()
    ) {
        handleEvents {
            onToolCall { tool: Tool<*,*>, toolArgs: ToolArgs ->
                println("\n--- Calling Tool: ${tool.name} ---")
                println("Args: $toolArgs")
                println("------------------------------------")
            }
            onAgentFinished { _, result ->
                println("\n====== Agent Response ======")
                println(result)
                println("==========================\n")
            }
        }
    }

    // --- REPL Start ---
    println("### Interactive EDA Agent ###")
    println("Commands: :add-data <path>, :add-context <path>, :list, :exit")

    while (true) {
        print("> ")
        val userInput = readlnOrNull()?.trim() ?: break

        when {
            userInput.equals(":exit", ignoreCase = true) -> break

            userInput.startsWith(":add-data ") -> {
                val path = userInput.substringAfter(":add-data ").trim()
                println(toolset.load_data(path))
            }

            userInput.startsWith(":add-context ") -> {
                val pathStr = userInput.substringAfter(":add-context ").trim()
                val file = File(pathStr)
                if (!file.exists()) {
                    println("Error: File or directory not found at '$pathStr'")
                } else {
                    agentContext.contextFiles[file.name] = file.absolutePath
                    println("Added context file: '${file.name}'")
                }
            }

            userInput.equals(":list", ignoreCase = true) -> {
                println(toolset.list_context())
            }

            else -> {
                if (userInput.isBlank()) continue

                // Dynamically build the system prompt with the latest context
                val dynamicSystemPrompt = buildString {
                    appendLine("You are an expert data analysis assistant.")
                    if (agentContext.dataFiles.isNotEmpty()) {
                        appendLine("\nThe following data has been loaded into pandas DataFrames:")
                        agentContext.dataFiles.forEach { (name, path) ->
                            appendLine("- A DataFrame named `$name` from file `$path`.")
                        }
                    }
                    if (agentContext.contextFiles.isNotEmpty()) {
                        appendLine("\nThe following context files are available for reading:")
                        agentContext.contextFiles.forEach { (name, _) ->
                            appendLine("- `$name`")
                        }
                        appendLine("Use the `read_file` tool to inspect their contents when necessary.")
                    }
                    appendLine("\nYour task is to answer the user's question. Use the available tools to perform your analysis.")
                }

                // Create a temporary agent with the updated prompt for this specific run
                val tempAgent = AIAgent(
                    executor = simpleGrazieExecutor(apiKey),
                    llmModel = JetBrainsAIModels.OpenAI.GPT4o,
                    systemPrompt = dynamicSystemPrompt,
                    toolRegistry = agent.toolRegistry,
                    strategy = edaStrategy()
                )
                tempAgent.run(userInput)
            }
        }
    }

    // --- Cleanup ---
    println("Exiting agent and shutting down kernel...")
    process.destroy()
    println("Session finished.")
}

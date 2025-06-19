package eda

import ai.jetbrains.code.prompt.executor.clients.grazie.koog.model.JetBrainsAIModels
import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.tools.Tool
import ai.koog.agents.core.tools.ToolArgs
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.core.tools.reflect.asTools
import ai.koog.agents.ext.agent.ProvideStringSubgraphResult
import ai.koog.agents.features.eventHandler.feature.handleEvents
import ai.koog.book.utils.simpleGrazieExecutor
import ai.koog.prompt.dsl.prompt
import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.system.exitProcess


private const val PYTHON_KERNEL_PATH = "eda/src/main/kernel.py"
const val GUIDELINES_PATH = "eda/src/main/guidelines.md"


fun main() = runBlocking {
    if (!File(PYTHON_KERNEL_PATH).exists()) {
        println("FATAL ERROR: Python kernel script not found at '$PYTHON_KERNEL_PATH'.")
        println("Please save the `kernel.py` file and update the PYTHON_KERNEL_PATH constant in `main.kt`.")
        exitProcess(1)
    }

    val process = ProcessBuilder("python", "-u", PYTHON_KERNEL_PATH).start()

    val kernelInput = process.outputStream
    val kernelOutput = process.inputStream.bufferedReader()

    Runtime.getRuntime().addShutdownHook(Thread {
        println("Shutting down Python kernel...")
        process.destroy()
    })

    val toolset = EdaTools(kernelInput, kernelOutput)
    val toolRegistry = ToolRegistry {
        tools(toolset.asTools())
        tool(ProvideStringSubgraphResult)
    }

    val apiToken = System.getenv("GRAZIE_TOKEN") ?: run {
        println("Error: GRAZIE_TOKEN environment variable not set.")
        exitProcess(1)
    }

    val agent = AIAgent(
        promptExecutor = simpleGrazieExecutor(apiToken),
        strategy = mainEdaStrategy(toolset),
        agentConfig = AIAgentConfig(
            prompt = prompt("eda-agent-base") {
                system("You are an AI assistant that helps users with data analysis.")
            },
            model = JetBrainsAIModels.OpenAI.GPT4_1,
            maxAgentIterations = 500
        ),
        toolRegistry = toolRegistry,
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

    println("### Interactive EDA Agent ###")
    println("You can now ask questions or give commands in natural language.")
    println("Examples: 'Load the dataset from ./data/sales.csv', 'what is the average price?', 'exit'")

    if (File(GUIDELINES_PATH).exists()) {
        println("Found guidelines file. Processing initial commands...")
        val guidelines = File(GUIDELINES_PATH).readText()
        try {
            agent.run(guidelines)
        } catch (e: Exception) {
            println("Error processing guidelines: ${e.message}")
            e.printStackTrace()
        }
    }

    while (true) {
        print("> ")
        val userInput = readlnOrNull()?.trim() ?: break

        if (userInput.equals("exit", ignoreCase = true)) {
            break
        }
        if (userInput.isBlank()) {
            continue
        }

        try {
            agent.run(userInput)
        } catch (e: Exception) {
            println("An unexpected error occurred: ${e.message}")
            e.printStackTrace()
        }
    }

    println("Exiting agent...")
    return@runBlocking
}

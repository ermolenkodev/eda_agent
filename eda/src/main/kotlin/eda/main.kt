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

fun main() = runBlocking {
    println("### Welcome to the EDA Agent ###")
    print("Please provide the path to your CSV file: ")
    val dataFilePath = readlnOrNull()?.trim()

    if (dataFilePath.isNullOrBlank()) {
        println("Error: No file path provided.")
        return@runBlocking
    }

    val dataFile = File(dataFilePath)
    if (!dataFile.exists() || !dataFile.isFile) {
        println("Error: File not found or is not a valid file at '$dataFilePath'")
        return@runBlocking
    }

    // TODO Provide the dataset's schema (column headers) to the LLM for context.
    val header = dataFile.bufferedReader().use { it.readLine() }
    val systemPrompt = """
        You are an expert data analysis assistant. 
        A pandas DataFrame named 'df' has been loaded with data from a CSV file.
        The file has the following columns: $header
        
        Your task is to answer the user's questions about this dataset.
        To do this, you MUST use the `pandas_executor` tool to write and execute Python code.
        Analyze the user's question, write the appropriate pandas code, and call the tool.
        
        If the tool returns an error, analyze the error and call the tool again with corrected code.
        
        When you have the final answer, provide a clear, natural language response to the user. Do not call any tools in your final response.
    """.trimIndent()

    val toolRegistry = ToolRegistry {
        tools(EdaTools(dataFilePath).asTools())
    }

    val apiKey = System.getenv("GRAZIE_TOKEN") ?: run {
        println("Error: OPENAI_API_KEY environment variable not set.")
        return@runBlocking
    }

    val agent = AIAgent(
        executor = simpleGrazieExecutor(apiKey),
        llmModel = JetBrainsAIModels.OpenAI.GPT4o,
        systemPrompt = systemPrompt,
        toolRegistry = toolRegistry,
        strategy = edaStrategy()
    ) {
        handleEvents {
            onToolCall { tool: Tool<*, *>, toolArgs: ToolArgs ->
                println("--- Calling Tool: ${tool.name} ---")
                val code = toolArgs.toString().substringAfter("code=").removeSuffix(")")
                println("------------------------------------")
            }
            onAgentFinished { _, result ->
                println("\n====== Agent Response ======")
                println(result)
                println("==========================\n")
            }
        }
    }

    while (true) {
        print("Ask a question about your data (or type 'exit' to quit): ")
        val userInput = readlnOrNull()
        if (userInput.isNullOrBlank() || userInput.equals("exit", ignoreCase = true)) {
            break
        }
        try {
            agent.run(userInput)
        } catch (e: Exception) {
            println("An unexpected error occurred: ${e.message}")
        }
    }

    println("EDA Agent session finished.")
}

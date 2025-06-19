package eda


import ai.koog.agents.core.tools.SimpleTool
import ai.koog.agents.core.tools.ToolArgs
import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import ai.koog.agents.core.tools.annotations.LLMDescription
import kotlinx.serialization.Serializable
import ai.koog.agents.core.agent.entity.AIAgentStrategy
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.*
import java.io.File


@Serializable
data class FinishTaskArgs(
    @LLMDescription("The final, user-facing answer or a confirmation message.")
    val finalAnswer: String
) : ToolArgs

object FinishTaskExecutionTool : SimpleTool<FinishTaskArgs>() {
    override val argsSerializer = FinishTaskArgs.serializer()
    override val descriptor = ToolDescriptor(
        name = "finish_task_execution",
        description = "Call this with the final answer when the entire user request is complete.",
        requiredParameters = listOf(
            ToolParameterDescriptor("final_answer", "The final answer to be presented to the user.", ToolParameterType.String)
        )
    )
    override suspend fun doExecute(args: FinishTaskArgs): String = args.finalAnswer
}


val agentStateKey = createStorageKey<AgentState>("agent_state")

fun mainEdaStrategy(edaTools: EdaTools): AIAgentStrategy = strategy("main_eda_strategy") {
    val initState by node<String, String>("initialize_or_get_state") { input ->
        storage.get(agentStateKey) ?: storage.set(agentStateKey, AgentStateHolder.state)
        input
    }

    val loadGuidelines by node<String, String>("load_guidelines") { input ->
        val guidelinesFile = File(GUIDELINES_PATH)
        if (guidelinesFile.exists()) {
            edaTools.addContext(GUIDELINES_PATH)
        }
        input
    }

    val decideNextAction by nodeLLMRequest("decideNextAction")
    val executeAction by nodeExecuteTool("executeAction")
    val sendResultAndLoop by nodeLLMSendToolResult("sendResultAndLoop")

    edge(nodeStart forwardTo initState)
    edge(initState forwardTo loadGuidelines)
    edge(loadGuidelines forwardTo decideNextAction transformed { userInput ->
        val agentState = storage.getValue(agentStateKey)
        buildString {
            appendLine("You are an expert data analyst and exploratory-data-analysis (EDA) assistant.")
            appendLine("Your mission is to satisfy the user's request: \"$userInput\".")
            appendLine("To do this, you can load data, inspect it, and execute Python code using the available tools.")

            appendLine("=== HOW TO WORK ===")
            appendLine("1. Check whether the request is about **data lineage / ETL logic** or about **the data itself**.")
            appendLine("   • If it's lineage-related, immediately scan the provided context/code files and extract the relevant facts without asking for user approval.")
            appendLine("   • If it's data-related, load the dataset(s) and use pandas-compatible Python code.")
            appendLine("2. Plan the *next* logical action before executing any code. If the goal or inputs are unclear, ask a clarifying question.")
            appendLine("3. When you run Python, ALWAYS show the final answer with `print()`—no print, no credit.")
            appendLine("4. Keep code blocks minimal: import only what you need, avoid side-effects, and label intermediate prints clearly if they're useful.")

            appendLine("=== BEST-PRACTICE REMINDERS ===")
            appendLine("• Handle missing or suspect values explicitly; note any assumptions.")
            appendLine("• For lineage questions, cite file names, function names, or line numbers so the user can trace your reasoning.")
            appendLine("• If an operation fails, diagnose briefly and suggest a fix rather than dumping stack traces.")

            appendLine("=== OUTPUT STYLE ===")
            appendLine("• Begin with a one-sentence answer, then provide supporting details.")
            appendLine("• Use markdown bullets or short tables for readability; avoid long prose.")
            appendLine("• End with your proposed *next action* if further steps are required.")

            appendLine("Now decide the next logical step and proceed.")

            if (agentState.contextFiles.containsKey("guidelines.md")) {
                appendLine("\nIMPORTANT: Guidelines for analysis is provided. Read them before proceeding. If it contains list of datasets load them with `loadData` tool. If it contains list of context files load them with `add_context` tool and immediately read them if the user's question is about lineage or ETL logic.")
            }

            if (agentState.dataFiles.isNotEmpty() || agentState.contextFiles.isNotEmpty()) {
                appendLine("\n### CURRENT STATE ###")
                if (agentState.dataFiles.isNotEmpty()) {
                    appendLine("The following dataframes are already loaded: ${agentState.dataFiles.keys.joinToString(", ")}")
                }
                if (agentState.contextFiles.isNotEmpty()) {
                    appendLine("The following context files are available: ${agentState.contextFiles.keys.joinToString(", ")}")
                }

                if (agentState.metadata.isNotBlank()) {
                    appendLine("\n### KNOWN DATAFRAME METADATA ###")
                    appendLine(agentState.metadata)
                    appendLine("\nUse this metadata to write your analysis code. Do NOT call `getDataframeInfo` for dataframes if their metadata is already listed above.")
                }

                appendLine("You can inspect them with `getDataframeInfo` or `read_file`.")
                appendLine("\n### ENHANCED FILE READING CAPABILITIES ###")
                appendLine("The `readFile` tool has been enhanced with specialized capabilities:")
                appendLine("- For large files (>700 lines), it provides intelligent file analysis with specialized guidance for Python and SQL files")
                appendLine("- You can search for specific terms or patterns using the `searchTerm` parameter")
                appendLine("- By default, the tool retrieves 1000 lines of context to provide comprehensive understanding")
                appendLine("- Variable context window extraction lets you control how many lines before/after matches are shown using `contextBefore` and `contextAfter` parameters (defaults: 50 lines before, 100 lines after)")
                appendLine("- For Python files: Get specialized guidance on functions, classes, data analysis patterns, pandas operations, etc.")
                appendLine("- For SQL files: Get specialized guidance on table definitions, queries, joins, filtering, aggregations, etc.")
                appendLine("- IMPORTANT: Prefer making fewer calls with larger context windows rather than many small calls")
                appendLine("Example usage: `readFile(\"script.py\", searchTerm=\"def process_data\")`")
            } else {
                appendLine("\nNo data is currently loaded.")
            }
        }
    })
    edge(decideNextAction forwardTo executeAction onToolCall { true })
    edge(executeAction forwardTo nodeFinish onCondition { it.tool == FinishTaskExecutionTool.name } transformed { it.content })
    edge(executeAction forwardTo sendResultAndLoop onCondition { it.tool != FinishTaskExecutionTool.name })
    edge(sendResultAndLoop forwardTo executeAction onToolCall { true })
    edge(sendResultAndLoop forwardTo nodeFinish onAssistantMessage { true })
    edge(decideNextAction forwardTo nodeFinish onAssistantMessage { true })
}

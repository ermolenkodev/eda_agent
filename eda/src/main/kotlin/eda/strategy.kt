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
import ai.koog.agents.core.environment.ReceivedToolResult
import ai.koog.agents.core.tools.reflect.asTool

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

    val decideNextAction by nodeLLMRequest("decideNextAction")
    val executeAction by nodeExecuteTool("executeAction")
    val sendResultAndLoop by nodeLLMSendToolResult("sendResultAndLoop")

    edge(nodeStart forwardTo initState)
    edge(initState forwardTo decideNextAction transformed { userInput ->
        val agentState = storage.getValue(agentStateKey)
        buildString {
            appendLine("You are an expert data analyst. Your goal is to answer the user's request: \"$userInput\"")
            appendLine("To do this, you can load data, inspect it, and execute Python code using the available tools.")
            appendLine("\nIMPORTANT: Your Python code MUST use the `print()` function to output the final result of any calculation or data lookup. Code that does not use `print()` will produce no output and the analysis will fail.")
            appendLine("\nOften user query is not about data itself but about the linage in that case careful analysis of context files is required to extract facts from the provided code or textual data.")
            appendLine("Decide the next logical step to achieve the user's goal.")

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
                    appendLine("\nUse this metadata to write your analysis code. Do NOT call `get_dataframe_info` for dataframes if their metadata is already listed above.")
                }

                appendLine("You can inspect them with `get_dataframe_info` or `read_file`.")
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
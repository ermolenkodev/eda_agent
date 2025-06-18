import ai.koog.agents.core.agent.entity.AIAgentStrategy
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.*
import ai.koog.agents.core.feature.AIAgentFeature
import ai.koog.agents.core.feature.AIAgentPipeline
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.reflect.asTool
import ai.koog.agents.ext.agent.ProvideStringSubgraphResult
import ai.koog.agents.ext.agent.StringSubgraphResult
import ai.koog.agents.ext.agent.subgraphWithTask
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.structure.json.JsonSchemaGenerator
import ai.koog.prompt.structure.json.JsonStructuredData
import kotlinx.serialization.Serializable

// --- Feature to Share AgentContext ---
class AgentContextFeature(val context: AgentContext) {
    companion object Feature : AIAgentFeature<AgentContext, AgentContextFeature> {
        override val key: AIAgentStorageKey<AgentContextFeature> = createStorageKey("agent_context_feature")
        override fun createInitialConfig() = AgentContext()
        override fun install(config: AgentContext, pipeline: AIAgentPipeline) {
            pipeline.interceptContextAgentFeature(this) { AgentContextFeature(config) }
        }
    }
}

// --- Serializable classes for Intent Classification ---
@Serializable
enum class UserIntent {
    LOAD_DATA,
    ADD_CONTEXT,
    LIST_CONTEXT,
    ANALYZE_DATA
}

@Serializable
data class ClassifiedIntent(
    @LLMDescription("The type of action the user wants to perform.")
    val intent: UserIntent,
    @LLMDescription("The file path provided by the user, if applicable.")
    val path: String? = null,
    @LLMDescription("The user's original question or analysis request, if applicable.")
    val question: String? = null
)

/**
 * A sophisticated strategy that uses subgraphs to manage different tasks.
 */
fun mainEdaStrategy(): AIAgentStrategy = strategy("main_eda_strategy") {

    // Subgraph 1: Classify the user's raw input into a structured intent.
    val classify by subgraph<String, ClassifiedIntent>("classify_intent") {
        val classifyNode by nodeLLMRequestStructured(
            structure = JsonStructuredData.createJsonStructure<ClassifiedIntent>(
                schemaFormat = JsonSchemaGenerator.SchemaFormat.JsonSchema,
                examples = listOf(
                    ClassifiedIntent(UserIntent.LOAD_DATA, "/path/to/data.csv"),
                    ClassifiedIntent(UserIntent.ANALYZE_DATA, question = "what is the average age?")
                )
            ),
            retries = 1,
            fixingModel = ai.koog.prompt.executor.clients.openai.OpenAIModels.Chat.GPT4o
        )
        edge(nodeStart forwardTo classifyNode)
        edge(classifyNode forwardTo nodeFinish transformed { it.getOrThrow().structure })
    }

    // Subgraph 2: The main analysis loop, modeled as a task.
    val analysis by subgraphWithTask<String, StringSubgraphResult>(
        tools = listOf(EdaTools::pandas_executor, EdaTools::read_file).map { it.asTool() },
        finishTool = ProvideStringSubgraphResult
    ) { question ->
        // This block defines the dynamic system prompt for the analysis task
        val agentContext = featureOrThrow(AgentContextFeature).context
        buildString {
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
            appendLine("\nYour task is to answer the following user question: \"$question\"")
            appendLine("Use the available tools to perform your analysis. When you have the final answer, call the `finish_task_execution` tool with the result.")
        }
    }

    // A simple node to execute a single tool call for context management.
    val executeContextTool by nodeExecuteTool("execute_context_tool")

    // --- Main Strategy Workflow ---

    // 1. Start by classifying the user's intent.
    edge(nodeStart forwardTo classify)

    // 2. Based on intent, route to the correct tool or subgraph.
    edge(classify forwardTo executeContextTool onToolCall { classifiedIntent ->
        when (classifiedIntent.intent) {
            UserIntent.LOAD_DATA -> tool.name == "load_data" && classifiedIntent.path?.let { args.path == it } ?: false
            UserIntent.ADD_CONTEXT -> tool.name == "add_context" && classifiedIntent.path?.let { args.path == it } ?: false
            UserIntent.LIST_CONTEXT -> tool.name == "list_context"
            else -> false
        }
    })

    edge(classify forwardTo analysis onAssistantMessage {
        it.intent == UserIntent.ANALYZE_DATA
    } transformed { it.question ?: "Perform analysis." })

    // 3. After a context tool or analysis subgraph finishes, the agent's turn is done.
    edge(executeContextTool forwardTo nodeFinish transformed { it.content })
    edge(analysis forwardTo nodeFinish transformed { (it as StringSubgraphResult).result })
}
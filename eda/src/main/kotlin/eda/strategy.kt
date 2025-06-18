package eda

import ai.koog.agents.core.agent.entity.AIAgentStrategy
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.nodeExecuteTool
import ai.koog.agents.core.dsl.extension.nodeLLMRequest
import ai.koog.agents.core.dsl.extension.nodeLLMSendToolResult
import ai.koog.agents.core.dsl.extension.onAssistantMessage
import ai.koog.agents.core.dsl.extension.onToolCall

/**
 * 1. Call the LLM to get code.
 * 2. Execute the code.
 * 3. Send the results (or errors) back to the LLM.
 * 4. The LLM can then either try generating new code (continuing the loop) or provide a final answer (exiting the loop).
 */
fun edaStrategy(): AIAgentStrategy = strategy("eda_strategy") {
    val nodeCallLLM by nodeLLMRequest("call_llm_for_code")
    val nodeExecutePandas by nodeExecuteTool("execute_pandas_code")
    val nodeSendResultToLLM by nodeLLMSendToolResult("send_tool_result_to_llm")

    edge(nodeStart forwardTo nodeCallLLM)

    edge(nodeCallLLM forwardTo nodeExecutePandas onToolCall { true })

    // If the LLM returns a plain text answer, the task is done.
    edge(nodeCallLLM forwardTo nodeFinish onAssistantMessage { true })

    // After executing the code, send the result back to the LLM.
    edge(nodeExecutePandas forwardTo nodeSendResultToLLM)

    // After the LLM sees the result, it might decide to call the tool again (e.g., to correct an error).
    edge(nodeSendResultToLLM forwardTo nodeExecutePandas onToolCall { true })

    // Or, if the result is good, it will formulate a final text answer and finish.
    // TODO distinguish good result from failed
    edge(nodeSendResultToLLM forwardTo nodeFinish onAssistantMessage { true })
}

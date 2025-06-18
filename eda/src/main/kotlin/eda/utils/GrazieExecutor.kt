package ai.koog.book.utils

import ai.grazie.api.gateway.client.SuspendableAPIGatewayClient
import ai.grazie.client.common.SuspendableClientWithBackoff
import ai.grazie.client.common.SuspendableHTTPClient
import ai.grazie.client.ktor.GrazieKtorHTTPClient
import ai.grazie.model.auth.GrazieAgent
import ai.grazie.model.auth.v5.AuthData
import ai.jetbrains.code.prompt.executor.clients.grazie.koog.GrazieLLMClient
import ai.jetbrains.code.prompt.executor.clients.grazie.koog.model.GrazieEnvironment
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor

fun simpleGrazieExecutor(token: String): SingleLLMPromptExecutor {
    val client = SuspendableAPIGatewayClient(
        GrazieEnvironment.Production.url,
        SuspendableHTTPClient.WithV5(
            SuspendableClientWithBackoff(
                GrazieKtorHTTPClient.Client.Default,
            ), AuthData(
                token,
                grazieAgent = GrazieAgent("koog-agents-workshop", "dev")
            )
        )
    )

    val executor = SingleLLMPromptExecutor(GrazieLLMClient(client))
    return executor
}
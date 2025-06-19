package eda

import ai.koog.agents.core.tools.ToolResult
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.File
import java.io.OutputStream

data class AgentState(
    val dataFiles: MutableMap<String, String> = mutableMapOf(), // var_name -> path
    val contextFiles: MutableMap<String, String> = mutableMapOf(),  // file_name -> path
    var metadata: String = ""
)

object AgentStateHolder {
    val state = AgentState()
}


@Serializable
sealed interface KernelCommand

@Serializable
data class LoadCommand(val path: String, val varName: String) : KernelCommand

@Serializable
data class ExecuteCommand(val code: String) : KernelCommand

@Serializable
data class LoadDataResult(val message: String, val variableName: String) : ToolResult.JSONSerializable<LoadDataResult> {
    override fun getSerializer() = serializer()
}

@Serializable
data class GetInfoCommand(val varName: String) : KernelCommand


@LLMDescription("Tools for managing data and context for Exploratory Data Analysis.")
class EdaTools(
    private val kernelInput: OutputStream,
    private val kernelOutput: BufferedReader
) : ToolSet {

    private val agentState = AgentStateHolder.state

    private fun sendCommandToKernel(command: KernelCommand): String {
        val jsonCommand = Json.encodeToString(command)
        kernelInput.write((jsonCommand + "\n").toByteArray())
        kernelInput.flush()

        val output = StringBuilder()
        while (true) {
            val line = kernelOutput.readLine() ?: break
            if (line == "<END_OF_OUTPUT>") {
                break
            }
            output.append(line).append("\n")
        }
        return output.toString().trim()
    }

    @Tool
    @LLMDescription("Loads a data file (CSV, Excel, JSON, Parquet) into a pandas DataFrame. Returns the name of the variable assigned to the DataFrame.")
    fun loadData(
        @LLMDescription("The absolute or relative path to the data file.")
        path: String
    ): String {
        val file = File(path)
        if (!file.exists()) {
            return "Error: File not found at path '$path'."
        }

        val variableName = file.nameWithoutExtension.replace(Regex("[^a-zA-Z0-9]"), "_") + "_df"
        val command = LoadCommand(path = file.absolutePath, varName = variableName)
        val result = sendCommandToKernel(command)

        if (result.startsWith("PYTHON_ERROR:")) {
            return result
        }

        agentState.dataFiles[variableName] = file.absolutePath

        return "Data from '$path' loaded into DataFrame variable: `$variableName`."
    }

    @Tool
    @LLMDescription("Adds a file or a folder to the agent's context. The agent can then read these files to understand data transformations or guidelines.")
    fun addContext(
        @LLMDescription("The absolute or relative path to the file or folder to add to the context.")
        path: String
    ): String {
        val file = File(path)
        if (!file.exists()) {
            return "Error: File or directory not found at '$path'"
        }

        val filesToAdd = if (file.isDirectory) {
            file.walk().filter { it.isFile }.toList()
        } else {
            listOf(file)
        }

        filesToAdd.forEach { f ->
            agentState.contextFiles[f.name] = f.absolutePath
        }

        return "Added ${filesToAdd.size} file(s) to context: ${filesToAdd.joinToString { it.name }}"
    }

    @Tool
    @LLMDescription("Retrieves metadata for a loaded DataFrame, such as column names, data types, and basic statistics.")
    fun getDataframeInfo(
        @LLMDescription("The name of the DataFrame variable to inspect (e.g., 'sales_df').")
        variableName: String
    ): String {
        if (variableName !in agentState.dataFiles) {
            return "Error: DataFrame `$variableName` not found. Use `list_context()` to see available DataFrames."
        }
        val command = GetInfoCommand(varName = variableName)

        val meta = sendCommandToKernel(command)
        val agentState = AgentStateHolder.state
        agentState.metadata += meta

        return meta
    }

    @Tool
    @LLMDescription("Executes Python code using pandas to analyze the loaded datasets. Use the DataFrame variables provided in the system prompt.")
    fun pandasExecutor(
        @LLMDescription("A string of valid Python code to be executed for data analysis.")
        code: String
    ): String {
        val command = ExecuteCommand(code = code)
        return sendCommandToKernel(command)
    }

    @Tool
    @LLMDescription("Lists all currently loaded data files and context files available to the agent.")
    fun listContext(): String {
        if (agentState.dataFiles.isEmpty() && agentState.contextFiles.isEmpty()) {
            return "The context is currently empty. Use natural language like 'load data from ...' or 'add context from ...' to add files."
        }
        val dataList = agentState.dataFiles.entries.joinToString("\n") { (name, path) -> "- DataFrame '$name' (from: $path)" }
        val contextList = agentState.contextFiles.entries.joinToString("\n") { (name, path) -> "- Context file '$name' (from: $path)" }
        return "Available DataFrames:\n$dataList\n\nAvailable Context Files:\n$contextList"
    }

    @Tool
    @LLMDescription("Reads the content of a specified context file that has been previously added.")
    fun readFile(
        @LLMDescription("The name of the context file to read (e.g., 'my_script.py', 'guidelines.md').")
        file_name: String
    ): String {
        val path = agentState.contextFiles[file_name]
            ?: return "Error: Context file '$file_name' not found. Use list_context() to see available files."
        return File(path).readText()
    }
}

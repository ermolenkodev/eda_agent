import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet
import ai.koog.agents.features.common.config.FeatureConfig
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.File
import java.io.OutputStream

// Represents the state of the agent's context, including loaded data and files.
data class AgentState(
    val dataFiles: MutableMap<String, String> = mutableMapOf(), // var_name -> path
    val contextFiles: MutableMap<String, String> = mutableMapOf()  // file_name -> path
) : FeatureConfig()


// --- Serializable Command Classes for Kernel Communication ---
@Serializable
sealed interface KernelCommand

@Serializable
data class LoadCommand(val path: String, val var_name: String) : KernelCommand

@Serializable
data class ExecuteCommand(val code: String) : KernelCommand

/**
 * A set of tools for the EDA agent to interact with the Python kernel and manage its context.
 * The tools now receive the AgentContext via the feature system, not the constructor.
 */
@LLMDescription("Tools for managing data and context for Exploratory Data Analysis.")
class EdaTools(
    private val kernelInput: OutputStream,
    private val kernelOutput: BufferedReader
) : ToolSet {

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

    // Note: The 'context' parameter will be provided by the agent's strategy nodes.
    // This makes the tools more modular.

    @Tool
    @LLMDescription("Loads a data file (CSV, Excel, JSON, Parquet) into a pandas DataFrame. Returns the name of the variable assigned to the DataFrame.")
    fun load_data(
        @LLMDescription("The absolute or relative path to the data file.")
        path: String,
        context: AgentContext // Context is now passed during execution
    ): String {
        val file = File(path)
        if (!file.exists()) {
            return "Error: File not found at path '$path'."
        }

        val variableName = file.nameWithoutExtension.replace(Regex("[^a-zA-Z0-9]"), "_") + "_df"
        context.dataFiles[variableName] = file.absolutePath

        val command = LoadCommand(path = file.absolutePath, var_name = variableName)
        val result = sendCommandToKernel(command)

        if (result.startsWith("PYTHON_ERROR:")) {
            context.dataFiles.remove(variableName)
            return result
        }

        return "Data from '$path' loaded into DataFrame variable: `$variableName`."
    }

    @Tool
    @LLMDescription("Adds a file or a folder to the agent's context. The agent can then read these files to understand data transformations or guidelines.")
    fun add_context(
        @LLMDescription("The absolute or relative path to the file or folder to add to the context.")
        path: String,
        context: AgentContext
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
            context.contextFiles[f.name] = f.absolutePath
        }

        return "Added ${filesToAdd.size} file(s) to context: ${filesToAdd.joinToString { it.name }}"
    }

    @Tool
    @LLMDescription("Executes Python code using pandas to analyze the loaded datasets. Use the DataFrame variables provided in the system prompt.")
    fun pandas_executor(
        @LLMDescription("A string of valid Python code to be executed for data analysis.")
        code: String
    ): String {
        val command = ExecuteCommand(code = code)
        return sendCommandToKernel(command)
    }

    @Tool
    @LLMDescription("Lists all currently loaded data files and context files available to the agent.")
    fun list_context(context: AgentContext): String {
        if (context.dataFiles.isEmpty() && context.contextFiles.isEmpty()) {
            return "The context is currently empty. Use natural language like 'load data from ...' or 'add context from ...' to add files."
        }
        val dataList = context.dataFiles.entries.joinToString("\n") { (name, path) -> "- DataFrame '$name' (from: $path)" }
        val contextList = context.contextFiles.entries.joinToString("\n") { (name, path) -> "- Context file '$name' (from: $path)" }
        return "Available DataFrames:\n$dataList\n\nAvailable Context Files:\n$contextList"
    }

    @Tool
    @LLMDescription("Reads the content of a specified context file that has been previously added.")
    fun read_file(
        @LLMDescription("The name of the context file to read (e.g., 'my_script.py', 'guidelines.md').")
        file_name: String,
        context: AgentContext
    ): String {
        val path = context.contextFiles[file_name]
            ?: return "Error: Context file '$file_name' not found. Use list_context() to see available files."
        return File(path).readText()
    }
}
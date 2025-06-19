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
import java.util.regex.Pattern

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
    @LLMDescription("""Reads the content of a specified context file that has been previously added. For large files (>700 lines), it provides a summary and lets you use your reasoning to decide what to search for.

    Tool API Details:
    - For small files (<700 lines): Returns the entire file content or the specified portion.
    - For large files (>700 lines): 
      1. Without search_term: Provides file analysis with specialized guidance for Python and SQL files, showing file beginning and end.
      2. With search_term: Searches for the term and extracts context around matches with customizable window size.

    Advanced Features:
    - Specialized analysis for Python and SQL files with domain-specific pattern suggestions
    - Variable context window extraction (control how many lines before/after matches are shown)
    - Intelligent pattern matching that finds related terms if exact match fails
    - Regular expression support for complex search patterns
    """)
    fun readFile(
        @LLMDescription("The name of the context file to read (e.g., 'my_script.py', 'guidelines.md').")
        fileName: String,
        @LLMDescription("The line number to start reading from (0-based index). Default is 0 (start of file).")
        offset: Int = 0,
        @LLMDescription("The maximum number of lines to read. Default is 1000 to allow for larger context windows. Use -1 to read all lines from offset to end of file.")
        limit: Int = 1000,
        @LLMDescription("Optional: A specific search term or pattern to look for in the file. For large files, this helps focus on relevant sections. Supports regular expressions.")
        searchTerm: String = "",
        @LLMDescription("Optional: Number of lines to show before each match when using search_term. Default is 50.")
        contextBefore: Int = 50,
        @LLMDescription("Optional: Number of lines to show after each match when using search_term. Default is 100.")
        contextAfter: Int = 100
    ): String {
        val path = agentState.contextFiles[fileName]
            ?: return "Error: Context file '$fileName' not found. Use list_context() to see available files."

        if (offset < 0) {
            return "Error: Offset cannot be negative."
        }

        if (limit < -1) {
            return "Error: Limit cannot be less than -1."
        }

        val fileSizeResult = checkFileLength(path)
        val lineCountMatch = Regex("Lines: (\\d+)").find(fileSizeResult)
        val lineCount = lineCountMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0

        if (lineCount <= 700 || offset > 0 || limit != -1) {
            return File(path).useLines { lines ->
                lines.drop(offset)
                    .let { if (limit >= 0) it.take(limit) else it }
                    .joinToString("\n")
            }
        }

        return if (searchTerm.isNotEmpty()) {
            searchAndExtractContext(fileName, path, searchTerm, contextBefore, contextAfter)
        } else {
            provideLargeFileGuidance(fileName, path)
        }
    }

    private fun provideLargeFileGuidance(fileName: String, path: String): String {
        val result = StringBuilder()
        result.append("This is a large file (>700 lines). I'll help you analyze it effectively.\n\n")

        val fileExtension = fileName.substringAfterLast('.', "")

        val beginning = File(path).useLines { lines ->
            lines.take(100).joinToString("\n")
        }

        val end = File(path).useLines { lines ->
            val linesList = lines.toList()
            if (linesList.size > 200) {
                linesList.takeLast(100).joinToString("\n")
            } else {
                ""
            }
        }

        result.append("=== File Analysis ===\n")
        when (fileExtension.lowercase()) {
            "py" -> result.append("This appears to be a Python file. Here's specialized guidance for Python analysis:\n\n" +
                    "Common Python patterns to search for:\n" +
                    "- Function definitions: `def function_name`\n" +
                    "- Class definitions: `class ClassName`\n" +
                    "- Method definitions: `def __init__` or other methods\n" +
                    "- Import statements: `import module` or `from module import`\n" +
                    "- Variable assignments: `variable_name =`\n" +
                    "- Decorators: `@decorator`\n" +
                    "- Context managers: `with` statements\n" +
                    "- List/dictionary comprehensions: `[x for x in]` or `{key: value for}`\n" +
                    "- Error handling: `try:` or `except`\n" +
                    "- Main execution block: `if __name__ == \"__main__\":`\n\n" +

                    "Data analysis specific patterns:\n" +
                    "- Pandas operations: `df.`, `.groupby`, `.apply`, `.merge`\n" +
                    "- NumPy operations: `np.`, `.array`, `.reshape`\n" +
                    "- Plotting: `plt.`, `.plot`, `.figure`\n" +
                    "- Data loading: `.read_csv`, `.read_sql`, `.to_csv`\n" +
                    "- Statistical functions: `.mean()`, `.std()`, `.describe()`\n\n" +

                    "For effective Python code exploration:\n" +
                    "1. First identify the main functions/classes\n" +
                    "2. Look for data processing pipelines\n" +
                    "3. Check for configuration variables at the top of the file\n" +
                    "4. Examine import statements to understand dependencies\n")

            "sql", "ddl", "dml" -> result.append("This appears to be a SQL file. Here's specialized guidance for SQL analysis:\n\n" +
                    "Common SQL patterns to search for:\n" +
                    "- Table definitions: `CREATE TABLE`\n" +
                    "- View definitions: `CREATE VIEW`\n" +
                    "- Index definitions: `CREATE INDEX`\n" +
                    "- Queries: `SELECT` statements\n" +
                    "- Data manipulation: `INSERT`, `UPDATE`, `DELETE`\n" +
                    "- Joins: `JOIN`, `LEFT JOIN`, `INNER JOIN`\n" +
                    "- Filtering: `WHERE` clauses\n" +
                    "- Aggregations: `GROUP BY`, `HAVING`\n" +
                    "- Common table expressions: `WITH` statements\n" +
                    "- Stored procedures: `CREATE PROCEDURE`\n" +
                    "- Functions: `CREATE FUNCTION`\n" +
                    "- Triggers: `CREATE TRIGGER`\n\n" +

                    "For effective SQL exploration:\n" +
                    "1. First identify table schemas and relationships\n" +
                    "2. Look for complex queries and understand their purpose\n" +
                    "3. Examine join conditions to understand data relationships\n" +
                    "4. Check for performance optimization patterns like indexes\n" +
                    "5. Look for transaction handling with `BEGIN`, `COMMIT`, `ROLLBACK`\n")

            else -> result.append("Consider searching for keywords related to your task or specific patterns in the file.\n")
        }

        result.append("\n=== Beginning of file (first 100 lines) ===\n")
        result.append("```\n")
        result.append(beginning)
        result.append("\n```\n\n")

        if (end.isNotEmpty()) {
            result.append("=== End of file (last 100 lines) ===\n")
            result.append("```\n")
            result.append(end)
            result.append("\n```\n\n")
        }

        result.append("=== How to proceed ===\n")
        result.append("To explore this file effectively, call readFile again with a specific search_term parameter based on what you're looking for.\n")
        result.append("You can also control the context window size around matches using context_before and context_after parameters.\n\n")
        result.append("Examples:\n")
        result.append("- Basic search: readFile(\"$fileName\", search_term = \"your search term\")\n")
        result.append("- With custom context window: readFile(\"$fileName\", search_term = \"your search term\", context_before = 15, context_after = 25)\n\n")
        result.append("You can use your reasoning to decide what to search for based on the file structure and your task requirements.\n")
        result.append("Regular expressions are supported for more complex search patterns.\n")
        result.append("For Python and SQL files, refer to the specialized guidance above to identify the most relevant patterns to search for.\n")

        return result.toString()
    }

    private fun searchAndExtractContext(
        fileName: String, 
        path: String, 
        searchTerm: String,
        contextBefore: Int = 10,
        contextAfter: Int = 20
    ): String {
        val result = StringBuilder()
        result.append("Searching for '$searchTerm' in $fileName:\n\n")

        val searchResult = searchContextWithRegex(searchTerm, fileName)

        if (searchResult.contains("No matches found")) {
            result.append("No exact matches found for '$searchTerm'. Let me try to find related content:\n\n")

            val relatedPatterns = generateRelatedPatterns(searchTerm)
            var foundRelated = false

            for (pattern in relatedPatterns) {
                val relatedResult = searchContextWithRegex(pattern, fileName)
                if (!relatedResult.contains("No matches found")) {
                    foundRelated = true
                    result.append("=== Related matches for pattern: $pattern ===\n")
                    result.append(relatedResult)
                    result.append("\n\n")

                    extractAndAppendContext(result, relatedResult, path, contextBefore, contextAfter)
                }
            }

            if (!foundRelated) {
                result.append("Could not find related content. Here's a summary of the file structure instead:\n\n")
                result.append(provideLargeFileGuidance(fileName, path))
            }
        } else {
            result.append(searchResult)
            result.append("\n\n")

            extractAndAppendContext(result, searchResult, path, contextBefore, contextAfter)

            result.append("\n=== Suggested next searches ===\n")
            result.append("Based on your search for '$searchTerm', you might also want to search for related terms or patterns.\n")
            result.append("Use your reasoning to identify relevant connections in the code and explore them with additional searches.\n")
        }

        return result.toString()
    }

    private fun extractAndAppendContext(
        result: StringBuilder, 
        searchResult: String, 
        path: String,
        contextBefore: Int = 10,
        contextAfter: Int = 20
    ) {
        val lineMatches = Regex("Line (\\d+):").findAll(searchResult)
        val lineNumbers = lineMatches.map { it.groupValues[1].toInt() }.toList()

        for (lineNum in lineNumbers) {
            val startLine = maxOf(0, lineNum - contextBefore)
            val contextLines = File(path).useLines { lines ->
                lines.drop(startLine)
                    .take(contextBefore + 1 + contextAfter)
                    .joinToString("\n")
            }

            result.append("=== Context around line $lineNum (window: $contextBefore lines before, $contextAfter lines after) ===\n")
            result.append("```\n")
            result.append(contextLines)
            result.append("\n```\n\n")
        }
    }

    private fun generateRelatedPatterns(searchTerm: String): List<String> {
        val patterns = mutableListOf<String>()

        patterns.add("\\b${Regex.escape(searchTerm)}\\b")

        val camelCaseParts = searchTerm.split(Regex("(?<=[a-z])(?=[A-Z])"))
        val snakeCaseParts = searchTerm.split("_")

        if (camelCaseParts.size > 1) {
            camelCaseParts.forEach { part -> 
                if (part.length > 3) patterns.add("\\b${Regex.escape(part)}\\b") 
            }
        }

        if (snakeCaseParts.size > 1) {
            snakeCaseParts.forEach { part -> 
                if (part.length > 3) patterns.add("\\b${Regex.escape(part)}\\b") 
            }
        }

        if (searchTerm.matches(Regex("[a-zA-Z_][a-zA-Z0-9_]*"))) {
            patterns.add("(def|function|fun|public|private|protected)\\s+${Regex.escape(searchTerm)}")
            patterns.add("\\b${Regex.escape(searchTerm)}\\s*\\(")
        }

        if (searchTerm.matches(Regex("[A-Z][a-zA-Z0-9_]*"))) {
            patterns.add("(class|interface|enum)\\s+${Regex.escape(searchTerm)}")
        }

        return patterns
    }

    private fun smartReadLargeFile(fileName: String, path: String): String {
        return provideLargeFileGuidance(fileName, path)
    }

    @Tool
    @LLMDescription("Checks the length of a file in bytes and number of lines.")
    fun checkFileLength(
        @LLMDescription("The absolute or relative path to the file to check.")
        path: String
    ): String {
        val file = File(path)
        if (!file.exists()) {
            return "Error: File not found at path '$path'."
        }

        val sizeInBytes = file.length()
        val lineCount = file.bufferedReader().use { reader ->
            reader.lines().count()
        }

        return "File: $path\nSize: $sizeInBytes bytes\nLines: $lineCount"
    }

    @Tool
    @LLMDescription("Searches through context files using a regular expression pattern and returns matching lines.")
    fun searchContextWithRegex(
        @LLMDescription("The regular expression pattern to search for.")
        pattern: String,
        @LLMDescription("Optional: The specific context file to search in. If not provided, searches all context files.")
        fileName: String = ""
    ): String {
        if (agentState.contextFiles.isEmpty()) {
            return "Error: No context files available. Use addContext() to add files first."
        }

        try {
            val regex = Pattern.compile(pattern)
            val results = StringBuilder()
            var totalMatches = 0

            val filesToSearch = if (fileName.isNotEmpty()) {
                val path = agentState.contextFiles[fileName]
                    ?: return "Error: Context file '$fileName' not found. Use listContext() to see available files."
                mapOf(fileName to path)
            } else {
                agentState.contextFiles
            }

            filesToSearch.forEach { (name, path) ->
                var fileMatches = 0
                val matchesInFile = StringBuilder()

                File(path).useLines { lines ->
                    lines.forEachIndexed { index, line ->
                        if (regex.matcher(line).find()) {
                            matchesInFile.append("  Line ${index + 1}: $line\n")
                            fileMatches++
                            totalMatches++
                        }
                    }
                }

                if (fileMatches > 0) {
                    results.append("File: $name ($fileMatches matches)\n")
                    results.append(matchesInFile)
                    results.append("\n")
                }
            }

            return if (totalMatches > 0) {
                "Found $totalMatches matches across ${filesToSearch.size} file(s):\n\n$results"
            } else {
                "No matches found for pattern '$pattern' in ${filesToSearch.size} file(s)."
            }
        } catch (e: Exception) {
            return "Error: Invalid regex pattern - ${e.message}"
        }
    }
}

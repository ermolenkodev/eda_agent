package eda

import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet
import java.io.File
import java.util.concurrent.TimeUnit

@LLMDescription("Tools for Exploratory Data Analysis with pandas")
class EdaTools(private val dataFilePath: String) : ToolSet {

    @Tool
    @LLMDescription("Executes Python code using pandas to analyze the loaded dataset. The pandas DataFrame is pre-loaded into a variable named 'df'. You must print the result of your analysis to stdout (e.g., `print(df.head())`).")
    fun pandas_executor(
        @LLMDescription("A string containing valid Python code to be executed for data analysis using the pandas library.")
        code: String
    ): String {
        val scriptFile = File.createTempFile("eda_script_", ".py")

        // TODO reuse loaded dataframe
        scriptFile.writeText("""
import pandas as pd
import sys

try:
    # Load the dataset from the provided file path
    df = pd.read_csv('$dataFilePath')
    
    # Execute the LLM-generated code
${code.prependIndent("    ")}
except Exception as e:
    # Print any execution errors to stderr
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
        """.trimIndent())

        try {
            val process = ProcessBuilder("python", scriptFile.absolutePath)
                .redirectErrorStream(true)
                .start()

            val output = process.inputStream.bufferedReader().readText()
            process.waitFor(20, TimeUnit.SECONDS)

            return if (process.exitValue() == 0) {
                output.ifBlank { "Code executed successfully without producing output." }
            } else {
                "Execution failed with error: $output"
            }
        } catch (e: Exception) {
            return "Failed to execute script: ${e.message}"
        } finally {
            scriptFile.delete()
        }
    }
}

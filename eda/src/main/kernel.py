import sys
import pandas as pd
import json
import io

dataframes = {}

def execute_code(code_to_exec):
    try:
        exec_globals = {
            'pd': pd,
            **dataframes
        }
        exec(code_to_exec, exec_globals)
    except Exception as e:
        print(f"PYTHON_ERROR: {e}", file=sys.stderr)

def main():
    for line in sys.stdin:
        try:
            command_data = json.loads(line)
            command_type = command_data.get("type")

            if "LoadCommand" in command_type:
                path = command_data.get("path")
                var_name = command_data.get("varName")
                try:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                    elif path.endswith('.xlsx'):
                        df = pd.read_excel(path)
                    elif path.endswith('.json'):
                        df = pd.read_json(path)
                    elif path.endswith('.parquet'):
                        df = pd.read_parquet(path)
                    else:
                        raise ValueError(f"Unsupported file type for: {path}")

                    dataframes[var_name] = df
                    print(f"Successfully loaded '{path}' into DataFrame '{var_name}'.")
                except Exception as e:
                    print(f"PYTHON_ERROR: Failed to load data from {path}. Error: {e}", file=sys.stderr)

            elif "ExecuteCommand" in command_type:
                code = command_data.get("code")
                execute_code(code)

            elif "GetInfoCommand" in command_type:
                var_name = command_data.get("varName")
                df = dataframes.get(var_name)
                if df is not None:
                    # The complex Python logic is now encapsulated here in the kernel
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    description_str = df.describe(include='all').to_string()
                    head_str = df.head().to_string()
                    print(f"--- METADATA FOR {var_name} ---\n\n**Info:**\n{info_str}\n**Descriptive Statistics:**\n{description_str}\n\n**First 5 Rows:**\n{head_str}")
                else:
                    print(f"PYTHON_ERROR: DataFrame '{var_name}' not found.", file=sys.stderr)

        except json.JSONDecodeError as e:
            print(f"PYTHON_ERROR: Invalid JSON received. {e}", file=sys.stderr)

        print("<END_OF_OUTPUT>", flush=True)

if __name__ == "__main__":
    main()

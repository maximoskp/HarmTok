import json

for file_name in ["results/basic_eval_result", \
                  "results/basic_eval_reg_result"]:
    # Load JSON data from file
    with open(file_name + '.json', "r") as f:
        data = json.load(f)

    # Extract tokenizers (column names) and metrics (row names)
    tokenizers = list(data.keys())  
    metrics = list(next(iter(data.values())).keys())  

    # Start LaTeX table
    latex_code = """
    \\begin{table}[h]
        \\centering
        \\renewcommand{\\arraystretch}{1.2}
        \\begin{tabular}{l""" + "c" * len(tokenizers) + """}
            \\hline
            \\textbf{Metric} & """ + " & ".join(f"\\textbf{{{tok}}}" for tok in tokenizers) + """ \\\\
            \\hline
    """

    # Fill in the table rows
    for metric in metrics:
        row_values = [f"{data[tok][metric]:.4f}" for tok in tokenizers]
        latex_code += f"        \\textbf{{{metric.replace('_', ' ')}}} & " + " & ".join(row_values) + " \\\\\n"

    # End LaTeX table
    latex_code += """        \\hline
        \\end{tabular}
        \\caption{Comparison of Tokenizer Performance on Various Metrics}
        \\label{tab:tokenizer_performance}
    \\end{table}
    """

    # Write LaTeX table to file
    with open(file_name + ".tex", "w") as f:
        f.write(latex_code)

    print(f"LaTeX table has been saved to {file_name}.tex")

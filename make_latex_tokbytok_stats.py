import pandas as pd
import json

def symbol_basic_eval(df, tokenizer_name='ChordSymbolTokenizer'):
    total_pieces = len(df['labels'])

    correct_bar_predictions = 0
    total_bars = 0

    correct_new_chord_predictions = 0
    total_new_chord_predictions = 0

    correct_position_predictions = 0
    total_position_predictions = 0

    correct_chord_predictions = 0
    correct_root_predictions = 0
    total_chord_predictions = 0

    for p_i in range(total_pieces):
        l = df['labels'].iloc[p_i]
        p = df['predictions'].iloc[p_i]
        # for each token that should have been predicted
        l_split = l.split(' ')
        p_split = p.split(' ')
        # arm root and type for RootType tokenizers
        arm_root = False
        tmp_root = None
        tmp_type = None
        # keep a chord buffer to accumulate elements of chords for 
        # non single-word chord representations
        l_chord_buffer = []
        p_chord_buffer = []
        i = 0
        while i < len (l_split):
            # how many bars were correctly predicted
            if l_split[i] == '<bar>':
                total_bars += 1
                arm_root = False
                if p_split[i] == '<bar>':
                    correct_bar_predictions += 1
            # how many new chords were correctly predicted
            if 'position_' in l_split[i]:
                total_new_chord_predictions += 1
                arm_root = True
                if 'position_' in p_split[i]:
                    correct_new_chord_predictions += 1
            # how many correct positions were predicted
            if 'position_' in l_split[i]:
                total_position_predictions += 1
                if p_split[i] == l_split[i]:
                    correct_position_predictions += 1
            # how many exact chords and roots were predicted
            if tokenizer_name == 'ChordSymbolTokenizer':
                if ':' in l_split[i]:
                    total_chord_predictions += 1
                    if p_split[i] == l_split[i]:
                        correct_chord_predictions += 1
                    l_chord_split = l_split[i].split(':')
                    p_chord_split = p_split[i].split(':')
                    if l_chord_split[0] == p_chord_split[0]:
                        correct_root_predictions += 1
            elif tokenizer_name == 'GCTSymbolTokenizer':
                if '[' in l_split[i]:
                    total_chord_predictions += 1
                    if p_split[i] == l_split[i]:
                        correct_chord_predictions += 1
                    l_chord_split = l_split[i][1:].split('x')
                    p_chord_split = p_split[i][1:].split('x')
                    if l_chord_split[0] == p_chord_split[0]:
                        correct_root_predictions += 1
            elif tokenizer_name == 'RootTypeTokenizer' or tokenizer_name == 'GCTRootTypeTokenizer':
                if arm_root:
                    total_chord_predictions += 1
                    # progress to root
                    i += 1
                    tmp_correct_root = False
                    if i < len(l_split):
                        if p_split[i] == l_split[i]:
                            correct_root_predictions += 1
                            tmp_correct_root = True
                    # progress to type
                    i += 1
                    if i < len(l_split):
                        if p_split[i] == l_split[i] and tmp_correct_root:
                            correct_chord_predictions += 1
            elif tokenizer_name == 'RootPCTokenizer' or tokenizer_name == 'GCTRootPCTokenizer':
                if arm_root:
                    total_chord_predictions += 1
                    # progress to root
                    i += 1
                    tmp_correct_root = False
                    if i < len(l_split):
                        if p_split[i] == l_split[i]:
                            correct_root_predictions += 1
                            tmp_correct_root = True
                    # progress to type
                    i += 1
                    while i < len(l_split):
                        if l_split[i] == '<bar>' or 'position_' in l_split[i]:
                            # already gone too far
                            i -= 1
                            break
                        l_chord_buffer.append( l_split[i] )
                        p_chord_buffer.append( p_split[i] )
                        i += 1
                    # check if type is the same
                    if set(l_chord_buffer).issubset( p_chord_buffer ) and tmp_correct_root:
                        correct_chord_predictions += 1
                    # reset buffers
                    l_chord_buffer = []
                    p_chord_buffer = []
            elif tokenizer_name == 'PitchClassTokenizer':
                if arm_root:
                    total_chord_predictions += 1
                    # progress to type
                    i += 1
                    while i < len(l_split):
                        if l_split[i] == '<bar>' or 'position_' in l_split[i]:
                            # already gone too far
                            i -= 1
                            break
                        l_chord_buffer.append( l_split[i] )
                        p_chord_buffer.append( p_split[i] )
                        i += 1
                    # check if type is the same
                    if set(l_chord_buffer).issubset( p_chord_buffer ):
                        correct_chord_predictions += 1
                    # reset buffers
                    l_chord_buffer = []
                    p_chord_buffer = []
            i += 1
    results = {
        'correct_bar_predictions': correct_bar_predictions/total_bars,
        'correct_new_chord_predictions': correct_new_chord_predictions/total_new_chord_predictions,
        'correct_position_predictions': correct_position_predictions/total_position_predictions,
        'correct_chord_predictions': correct_chord_predictions/total_chord_predictions,
        'correct_root_predictions': correct_root_predictions/total_chord_predictions
    }
    return results
# end symbol_basic_eval

tokenizers = {
    'ChordSymbolTokenizer': symbol_basic_eval,
    'RootTypeTokenizer': symbol_basic_eval,
    'PitchClassTokenizer': symbol_basic_eval,
    'RootPCTokenizer': symbol_basic_eval,
    # 'GCTRootPCTokenizer': symbol_basic_eval,
    # 'GCTSymbolTokenizer': symbol_basic_eval,
    # 'GCTRootTypeTokenizer': symbol_basic_eval
}

tokenized_folder = 'tok_by_tok/gpt/'

results = {}

for tokenizer_name in tokenizers.keys():
    if tokenizers[tokenizer_name] is not None:
        df = pd.read_csv( tokenized_folder + tokenizer_name + '.csv' )
        results[tokenizer_name] = tokenizers[tokenizer_name](df, tokenizer_name=tokenizer_name)

print(results['ChordSymbolTokenizer'])
# print(results['GCTSymbolTokenizer'])
print(results['RootTypeTokenizer'])
# print(results['GCTRootTypeTokenizer'])
print(results['RootPCTokenizer'])
# print(results['GCTRootPCTokenizer'])
print(results['PitchClassTokenizer'])

print(results)
with open('results/tokbytok_eval_gpt.json', 'w') as fp:
    json.dump(results, fp)

tokenized_folder = 'tok_by_tok/bart/'

results = {}

for tokenizer_name in tokenizers.keys():
    if tokenizers[tokenizer_name] is not None:
        df = pd.read_csv( tokenized_folder + tokenizer_name + '.csv' )
        results[tokenizer_name] = tokenizers[tokenizer_name](df, tokenizer_name=tokenizer_name)

print(results)
with open('results/tokbytok_eval_bart.json', 'w') as fp:
    json.dump(results, fp)

for file_name in ["results/tokbytok_eval_gpt", \
                  "results/tokbytok_eval_bart"]:
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

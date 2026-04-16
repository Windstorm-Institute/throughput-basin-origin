#!/usr/bin/env python3
"""
Windstorm Institute Paper 7 - Results Synthesis
Populate PAPER7_MASTER_SUMMARY.md with actual results from all experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASEDIR = Path('/home/user1-gpu/agi-extensions')

def load_exp3_results():
    """Load Experiment 3 results"""
    exp3_dir = BASEDIR / 'exp-3' / 'results'

    results = {}

    try:
        df_bpt = pd.read_csv(exp3_dir / 'exp3_bpt_comparison.csv')
        df_stats = pd.read_csv(exp3_dir / 'exp3_statistics.csv')
        df_seven = pd.read_csv(exp3_dir / 'exp3_seven_corpus.csv')

        # Calculate key metrics
        transformer_mean = df_bpt[df_bpt['arch_type'] == 'transformer']['bpt_mean'].mean()
        serial_mean = df_bpt[df_bpt['arch_type'] == 'serial']['bpt_mean'].mean()

        transformer_var = df_seven[df_seven['arch_type'] == 'transformer']['bpt'].var()
        serial_var = df_seven[df_seven['arch_type'] == 'serial']['bpt'].var()

        # Find statistical test results
        welch_test = df_stats[df_stats['test'].str.contains("Welch", na=False)]
        levene_test = df_stats[df_stats['test'].str.contains("Levene", na=False)]

        results['transformer_mean'] = transformer_mean
        results['serial_mean'] = serial_mean
        results['difference'] = abs(transformer_mean - serial_mean)
        results['transformer_var'] = transformer_var
        results['serial_var'] = serial_var

        if not welch_test.empty:
            results['welch_p'] = welch_test.iloc[0]['p_value']

        if not levene_test.empty:
            results['levene_p'] = levene_test.iloc[0]['p_value']

        logger.info("Loaded Experiment 3 results")

    except Exception as e:
        logger.error(f"Failed to load Experiment 3 results: {e}")
        results = None

    return results


def load_exp2_results():
    """Load Experiment 2 results"""
    exp2_dir = BASEDIR / 'exp-2' / 'results'

    results = {}

    try:
        df = pd.read_csv(exp2_dir / 'exp2_quantization.csv')

        # Find the cliff for each model
        precision_order = ['fp16', 'int8', 'int4', 'int3', 'int2']

        # Best bits per joule
        best_idx = df['bits_per_joule'].idxmax()
        best_config = df.loc[best_idx]

        results['best_config_model'] = best_config['model']
        results['best_config_precision'] = best_config['precision']
        results['best_bits_per_joule'] = best_config['bits_per_joule']
        results['best_bpt'] = best_config['bpt']

        # Find where BPT exceeds basin (> 6 bits)
        collapsed = df[df['bpt'] > 6.0]

        if not collapsed.empty:
            # Find minimum precision where collapse occurs
            precision_nums = {'fp16': 16, 'int8': 8, 'int4': 4, 'int3': 3, 'int2': 2}
            collapsed['prec_num'] = collapsed['precision'].map(precision_nums)
            cliff_precision = collapsed.loc[collapsed['prec_num'].idxmin(), 'precision']
            results['cliff_precision'] = cliff_precision
        else:
            results['cliff_precision'] = 'None (all in basin)'

        logger.info("Loaded Experiment 2 results")

    except Exception as e:
        logger.error(f"Failed to load Experiment 2 results: {e}")
        results = None

    return results


def load_exp6_results():
    """Load Experiment 6 results"""
    exp6_dir = BASEDIR / 'exp-6' / 'results'

    results = {}

    try:
        df = pd.read_csv(exp6_dir / 'exp6_energy.csv')

        # Best phi
        best_phi_idx = df['phi'].idxmin()
        best_phi = df.loc[best_phi_idx]

        results['best_phi_model'] = best_phi['model']
        results['best_phi_config'] = best_phi['config']
        results['best_phi'] = best_phi['phi']
        results['best_log10_phi'] = best_phi['log10_phi']

        # Best bits per joule
        df['bits_per_joule'] = 1.0 / df['energy_per_bit_J']
        best_bpj_idx = df['bits_per_joule'].idxmax()
        best_bpj = df.loc[best_bpj_idx]

        results['best_bpj_model'] = best_bpj['model']
        results['best_bpj_config'] = best_bpj['config']
        results['best_bpj'] = best_bpj['bits_per_joule']
        results['best_energy_per_bit'] = best_bpj['energy_per_bit_J']

        # Gaps to ribosome and Landauer
        RIBOSOME_PHI = 1.02
        RIBOSOME_ENERGY_PER_BIT = 3.78e-20

        avg_temp_K = df['temp_K'].mean()
        K_B = 1.380649e-23
        E_landauer = K_B * avg_temp_K * np.log(2)

        results['gap_to_ribosome_phi'] = best_phi['phi'] / RIBOSOME_PHI
        results['gap_to_ribosome_energy'] = best_bpj['energy_per_bit_J'] / RIBOSOME_ENERGY_PER_BIT
        results['gap_to_landauer'] = best_bpj['energy_per_bit_J'] / E_landauer

        results['oom_to_ribosome'] = np.log10(results['gap_to_ribosome_phi'])
        results['oom_to_landauer'] = np.log10(results['gap_to_landauer'])

        logger.info("Loaded Experiment 6 results")

    except Exception as e:
        logger.error(f"Failed to load Experiment 6 results: {e}")
        results = None

    return results


def load_exp1_results():
    """Load Experiment 1 results"""
    exp1_dir = BASEDIR / 'exp-1' / 'results'

    results = {}

    try:
        # Load corpus entropy
        df_entropy = pd.read_csv(exp1_dir / 'corpus_entropy.csv')

        # Load self-eval
        df_self = pd.read_csv(exp1_dir / 'exp1_self_eval.csv')

        # Merge to compare
        df_merged = df_self.merge(df_entropy, left_on='corpus', right_on='corpus')

        # Key test: SYN-8
        syn8_row = df_merged[df_merged['corpus'] == 'syn8']

        if not syn8_row.empty:
            syn8_target = syn8_row.iloc[0]['target_entropy']
            syn8_measured = syn8_row.iloc[0]['bpt']
            syn8_match = abs(syn8_target - syn8_measured) < 1.0  # Within 1 bit

            results['syn8_target'] = syn8_target
            results['syn8_measured'] = syn8_measured
            results['syn8_match'] = syn8_match
        else:
            results['syn8_target'] = np.nan
            results['syn8_measured'] = np.nan
            results['syn8_match'] = False

        # Check if models compress on WikiText
        df_wiki = pd.read_csv(exp1_dir / 'exp1_wikitext.csv')
        df_wiki = df_wiki[df_wiki['tokenizer'] == 'gpt2']

        if not df_wiki.empty:
            # Do high-entropy models compress toward ~4 on natural language?
            syn8_wiki = df_wiki[df_wiki['model'] == 'syn8']
            if not syn8_wiki.empty:
                results['syn8_wikitext_bpt'] = syn8_wiki.iloc[0]['bpt']
                results['syn8_compressed'] = abs(results['syn8_wikitext_bpt'] - 4.16) < 1.0
            else:
                results['syn8_wikitext_bpt'] = np.nan
                results['syn8_compressed'] = None

        # Layer entropy probe - does entropy compress?
        df_layer = pd.read_csv(exp1_dir / 'exp1_layer_entropy.csv')
        syn8_layers = df_layer[df_layer['model'] == 'syn8']

        if not syn8_layers.empty and len(syn8_layers) > 6:
            input_entropy = syn8_layers[syn8_layers['layer'] == 0]['entropy_bits'].values[0]
            middle_entropy = syn8_layers[syn8_layers['layer'] == 6]['entropy_bits'].values[0]

            results['syn8_input_entropy'] = input_entropy
            results['syn8_middle_entropy'] = middle_entropy
            results['syn8_entropy_compression'] = input_entropy - middle_entropy
        else:
            results['syn8_input_entropy'] = np.nan
            results['syn8_middle_entropy'] = np.nan
            results['syn8_entropy_compression'] = np.nan

        logger.info("Loaded Experiment 1 results")

    except Exception as e:
        logger.error(f"Failed to load Experiment 1 results: {e}")
        results = None

    return results


def determine_verdict(exp1_results, exp3_results):
    """Determine the main verdict: Data vs Architecture vs Physics"""

    evidence_data = []
    evidence_architecture = []
    evidence_physics = []

    # Experiment 1 evidence
    if exp1_results and not pd.isna(exp1_results.get('syn8_measured')):
        if exp1_results['syn8_match']:
            evidence_data.append("Exp1: SYN-8 model achieves ~8 BPT on 8-bit corpus (data-driven)")
        else:
            if exp1_results.get('syn8_compressed'):
                evidence_architecture.append("Exp1: SYN-8 model compresses to ~4 BPT despite 8-bit training (architecture-driven)")

        if not pd.isna(exp1_results.get('syn8_entropy_compression')):
            if exp1_results['syn8_entropy_compression'] > 2.0:
                evidence_architecture.append("Exp1: Layer entropy compresses >2 bits toward basin (architecture-driven)")
            else:
                evidence_data.append("Exp1: Layer entropy remains high (data-driven)")

    # Experiment 3 evidence
    if exp3_results:
        if exp3_results.get('serial_var', float('inf')) < exp3_results.get('transformer_var', 0):
            evidence_architecture.append("Exp3: Serial models show tighter variance (architecture-driven)")
        else:
            evidence_data.append("Exp3: No architecture effect on basin tightness")

    # Determine main verdict
    if len(evidence_architecture) > len(evidence_data):
        verdict = "ARCHITECTURE-DRIVEN"
        explanation = "The throughput basin appears to be primarily imposed by serial decoding architectures."
    elif len(evidence_data) > len(evidence_architecture):
        verdict = "DATA-DRIVEN"
        explanation = "The throughput basin appears to be primarily inherited from training data statistics."
    else:
        verdict = "COMBINATION"
        explanation = "The throughput basin emerges from both data constraints and architectural properties."

    return verdict, explanation, evidence_data, evidence_architecture, evidence_physics


def generate_synthesis():
    """Generate the complete synthesis"""
    logger.info("="*80)
    logger.info("SYNTHESIZING PAPER 7 RESULTS")
    logger.info("="*80)

    # Load all results
    exp1 = load_exp1_results()
    exp2 = load_exp2_results()
    exp3 = load_exp3_results()
    exp6 = load_exp6_results()

    # Determine verdict
    verdict, explanation, evidence_data, evidence_arch, evidence_phys = determine_verdict(exp1, exp3)

    # Generate summary text
    summary = []
    summary.append(f"\n## SYNTHESIS COMPLETE\n")
    summary.append(f"**Verdict:** {verdict}")
    summary.append(f"\n{explanation}\n")

    summary.append("\n### Evidence for Data-Driven Basin:")
    if evidence_data:
        for e in evidence_data:
            summary.append(f"- {e}")
    else:
        summary.append("- None")

    summary.append("\n### Evidence for Architecture-Driven Basin:")
    if evidence_arch:
        for e in evidence_arch:
            summary.append(f"- {e}")
    else:
        summary.append("- None")

    # Print to console
    print('\n'.join(summary))

    # Save to file
    output_file = BASEDIR / 'SYNTHESIS_SUMMARY.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary))

    logger.info(f"\nSynthesis saved to {output_file}")
    logger.info("\nNext step: Manually update PAPER7_MASTER_SUMMARY.md with these findings")


if __name__ == '__main__':
    generate_synthesis()

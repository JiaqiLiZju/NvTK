import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy


def get_activate_index_from_fmap(fmap, X, threshold=0.99):
    motif_nb = fmap.shape[1]
    X_dim, seq_len = X.shape[1], X.shape[-1]

    W={}
    for filter_index in range(motif_nb):
        # find regions above threshold
        data_index, pos_index = np.where(fmap[:,filter_index,:] > np.max(fmap[:,filter_index,:], axis=1, keepdims=True)*threshold)
        # TODO select gene instead of base-pair, whether axis=1 is ok?
        W[filter_index] = [data_index, pos_index]

    return W


def get_activate_regulon_fromW(W, gene_name=None, threshold=0.99, device=torch.device("cuda")):
    # fmap, X = get_fmap(model, hook_module, data, device=device)
    # W = get_activate_index_from_fmap(fmap, X, threshold=threshold)
    
    regulon, regulon_pos = {}, {}
    for k,v in W.items():
        gene_idx, pos_idx = deepcopy(v[0]), deepcopy(v[1])
        mask = (4500<pos_idx) & (pos_idx<8500) # -2k < pos < 2k
        gene_idx = gene_idx[mask]
        pos_idx = pos_idx[mask]

        gene_id, cnt = np.unique(gene_idx, return_counts=True)
        t = np.mean(cnt) + 3 * np.std(cnt)
        # t = np.quantile(cnt, 0.99)
        # t = np.max(cnt) * 0.4
        idx = np.where(cnt >= t)
        target_gene_id = gene_id[idx]
        regulon[k] = gene_name[target_gene_id]

        target_mask = [g in target_gene_id for g in gene_idx]
        target_gene = gene_idx[target_mask]
        target_pos = pos_idx[target_mask]
        regulon_pos[k] = list(zip(gene_name[target_gene], target_pos))
    
    motif, target, pos = [], [], []
    for k,v in regulon_pos.items():
        for t,p in v:
            motif.append(k)
            target.append(t)
            pos.append(p)
    regulon_pos_df = pd.DataFrame({"Filter": motif, "target": target, "position":pos})

    return regulon, regulon_pos, regulon_pos_df


def prune_regulon_by_expr_influe_correlation(regulon, influe, expr_adata, threshold=0.1):
    motif, target, corr = [], [], []
    for f,v in tqdm(regulon.items()):
        influe_filter = influe.iloc[f,:].values
        
        for t in v:
            motif.append(f)
            target.append(t)
            
            expr_gene = expr_adata[:,t].X.flatten()
            corr.append(pearsonr(influe_filter, expr_gene)[0])

    regulon_df = pd.DataFrame({"Filter": motif, "target": target, "Corr": corr}).fillna(0)
    pruned_regulon_df = regulon_df[regulon_df.Corr > threshold]

    return regulon_df, pruned_regulon_df


def calculate_filter_pairs_jaccard(regulon_df):
    filter_nb = 128
    filter_pairs = list(itertools.combinations(range(filter_nb), 2))

    f1_l, f2_l, j_l = [], [], []
    for f1, f2 in tqdm(filter_pairs):
        t1 = set(regulon_df[regulon_df.Filter==f1].target.values)
        t2 = set(regulon_df[regulon_df.Filter==f2].target.values)

        inter = t1 & t2
        outer = t1 | t2

        jaccard_score = len(inter) / len(outer)
        
        f1_l.append(f1)
        f2_l.append(f2)
        j_l.append(jaccard_score)

    filter_pairs_jaccard = pd.DataFrame({"Filter1":f1_l, "Filter2":f2_l, "Jaccard":j_l})
    # filter_pairs_jaccard["Pair"] = '(' + filter_pairs_jaccard.Filter1.astype("str") + "," + filter_pairs_jaccard.Filter2.astype("str") + ')'

    return filter_pairs_jaccard


def trim_duplicate_filter_pairs(filter_pairs_jaccard):
    tomtom = pd.read_csv("../../../Cross_Species_Motif/DD/tomtom.tsv", sep='\t', header=0, skipfooter=3, engine='python')
    tomtom = tomtom[["Query_ID", "Target_ID", "q-value"]]
    tomtom.columns = ["Filter1", "Filter2", "q_value"]
    tomtom['Pair'] = '(' + tomtom.Filter1.map(lambda x:x.split('_')[-1]) + ',' + tomtom.Filter2.map(lambda x:x.split('_')[-1]) + ')'
    mask = [pair not in tomtom.Pair.values for pair in filter_pairs_jaccard.Pair.values]
    # np.sum(mask)
    filter_pairs_jaccard_nonRedundant = filter_pairs_jaccard[mask]
    return filter_pairs_jaccard_nonRedundant


def trim_duplicate_filters():
    tomtom = pd.read_csv("../../Cross_Species_Motif/DD/tomtom.tsv", sep='\t', header=0, skipfooter=3, engine='python')
    tomtom = tomtom[["Query_ID", "Target_ID", "q-value"]]
    tomtom.columns = ["Filter1", "Filter2", "q_value"]
    tomtom['Pair'] = '(' + tomtom.Filter1.map(lambda x:x.split('_')[-1]) + ',' + tomtom.Filter2.map(lambda x:x.split('_')[-1]) + ')'
    tomtom = tomtom[tomtom["q_value"] < 0.01]

    IC = pd.read_csv("./Motif_filter/meme_conv1_thres9_IC_freq.csv", index_col=0)
    IC.index = IC.index.map(lambda x: "Motif_"+str(x))
    
    tomtom_ic = tomtom.merge(IC, left_on="Filter2", right_index=True, sort=False, how="left")
    nondup_filter = tomtom_ic[tomtom_ic["rank"]==1]
    return nondup_filter.Filter2.unique()


def regulon_combination(filter_pairs_jaccard_nonRedundant):
    t = np.mean(filter_pairs_jaccard_nonRedundant.Jaccard) + 3 * np.std(filter_pairs_jaccard_nonRedundant.Jaccard)
    filter_pairs_jaccard_nonRedundant = filter_pairs_jaccard_nonRedundant[filter_pairs_jaccard_nonRedundant.Jaccard > t]
    # filter_pairs_jaccard_nonRedundant.to_csv("./Motif/filter_pairs_jaccard_nonRedundant.pos2k.mean+3std.csv")
    return filter_pairs_jaccard_nonRedundant


def prune_regulonComb_by_expr_influeComb_correlation(filter_pairs, regulon_df, influe_comb, expr_adata, threshold=0.1):
    celltype, motif, target, corr = [], [], [], []
    for c in anno.Cellcluster.unique():
        c_idx = anno.Cellcluster == c
        
        for f1, f2, f in tqdm(filter_pairs[["Filter1", "Filter2", "Pair"]].values):

            t1 = regulon_df[regulon_df.Filter==f1].target.values
            t2 = regulon_df[regulon_df.Filter==f2].target.values
            t_inter = np.intersect1d(t1, t2)

            influe_comb_filter = influe_comb.loc[f, c_idx].values

            for t in t_inter:
                celltype.append(c)
                motif.append(f)
                target.append(t)

                expr_gene = expr[c_idx, t].X.flatten()
                corr.append(pearsonr(influe_comb_filter, expr_gene)[0])

    return regulon_df, pruned_regulon_df


def regulon_combination_pos(filter_pairs_jaccard_nonRedundant, regulon_pos_df):
    raise NotImplementedError

from Lib_SSL_CL import *


""" load dataset """
workflow = ["1000genome_new_2022", "montage"]
dat = []
for work in workflow:

    print(work)
    ROOT = osp.join(osp.expanduser("~"), "tmp", "data_new", work)
    pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                                T.ToUndirected(),
                                T.RandomNodeSplit(split="train_rest",
                                                    num_val=0.2,
                                                    num_test=0.2)])

    dataset = PSD_Dataset(root=ROOT,
                            name=work,
                            node_level=True,
                            binary_labels=True,
                            normalize=False,
                            pre_transform=pre_transform)
    
    print(len(dataset[0]) )
    dat.append(dataset[0])



dataTot = Merge_PSD_Dataset_v1(root=ROOT,
                            node_level=True,
                            binary_labels=True,
                            normalize=False,
                            pre_transform=pre_transform)

Tot = dataTot[0]

n_epoch = 100
n_mod = 5

dict = {}
for (data, work) in zip(dat, workflow):
    auc, ap, prec, rec = [], [], [], []
    for _ in range(n_mod):
        model = SSL(hid_dim=64,
                    weight_decay=1e-5,
                    dropout=0.5,
                    lr=1e-3,
                    epoch=n_epoch,
                    gpu=0,
                    alpha=0.5,
                    batch_size=32,
                    num_neigh=5,
                    verbose=True)

    
        model.fit(data, data.y)
        score = model.decision_scores_

        y = data.y.bool()
        k = sum(y)
        if np.isnan(score).any():
            warnings.warn('contains NaN, skip one trial.')
            # continue

        auc.append(eval_roc_auc(y, score))
        ap.append(eval_average_precision(y, score))
        prec.append(eval_precision_at_k(y, score, k))
        rec.append(eval_recall_at_k(y, score, k))

    print(f"{work}",
        f"{model.__class__.__name__:<15}",
        f"AUC: {np.mean(auc):.3f}±{np.std(auc):.3f} ({np.max(auc):.3f})",
        f"AP: {np.mean(ap):.3f}±{np.std(ap):.3f} ({np.max(ap):.3f})",
        f"Prec(K) {np.mean(prec):.3f}±{np.std(prec):.3f} ({np.max(prec):.3f})",
        f"Recall(K): {np.mean(rec):.3f}±{np.std(rec):.3f} ({np.max(rec):.3f})")
    dict[work] = (auc, ap, prec, rec)

with open('individual.npy', 'wb') as f:
    np.save(f, dict)
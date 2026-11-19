#!/usr/bin/env python3
"""Read-only HEART chemistry evidence utilities (external benchmarks + 10K similarity).

Original HEART model weights and original LMDB are NEVER modified. Data saved to new
work directories. No Slurm commands. External model inference must be explicitly invoked.
"""
from __future__ import annotations
import csv, hashlib, json, os, pickle, math, urllib.request, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

BENCHMARKS = {
 'ESOL': {
  'url':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
  'file':'delaney-processed.csv', 'target':'measured log solubility in mols per litre', 'label':'Experimental log solubility (log mol/L)'},
 'FreeSolv': {
  'url':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
  'file':'SAMPL.csv','target':'expt','label':'Experimental hydration free energy (kcal/mol)'},
 'Lipophilicity': {
  'url':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
  'file':'Lipophilicity.csv','target':'exp','label':'Experimental lipophilicity (logD)'},
}

def save_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,default=str)+'\n')

def get_benchmark_csvs(data_dir, try_download=False):
    data_dir=Path(data_dir);data_dir.mkdir(parents=True,exist_ok=True)
    missing=[]
    for name, spec in BENCHMARKS.items():
        dest=data_dir/spec['file']
        if not dest.is_file() or dest.stat().st_size<100:
            if try_download:
                try:
                    print('Fetching',name,'from original DeepChem mirror ...',flush=True)
                    urllib.request.urlretrieve(spec['url'],dest)
                except Exception as e:
                    dest.unlink(missing_ok=True)
                    print('Download unavailable:',str(e)[:200])
            if not dest.is_file(): missing.append((name,spec['url'],str(dest)))
    return missing

def molecules_from_benchmarks(data_dir,out_dir,max_atoms=256):
    """Standardize SMILES, remove benchmark exact duplicates, generate 1 seeded ETKDG geometry.

    One LMDB 'valid' with numeric ASCII keys, matching task.load_dataset('valid').
    Input labels NEVER enter model features. Missing RDKit geometries are skipped.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    import lmdb
    RDLogger.DisableLog('rdApp.warning')
    data_dir=Path(data_dir);out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=out_dir/'external_manifest.csv'
    lmdb_path=out_dir/'valid.lmdb'
    dict_path=out_dir/'dict.txt'
    if manifest_path.exists() or lmdb_path.exists():
        raise FileExistsError('External manifest/LMDB exists. Reuse it; do not overwrite or regenerate molecules. '+str(out_dir))
    rows=[]; seen=defaultdict(set);counts=Counter()
    for name,spec in BENCHMARKS.items():
        p=data_dir/spec['file'];raw=pd.read_csv(p)
        required=['smiles',spec['target']]
        if not all(c in raw.columns for c in required):
            raise ValueError(f'{p} columns mismatch; needed {required}; got {raw.columns.tolist()}')
        for _,item in raw.iterrows():
            smi=str(item['smiles']).strip(); y=pd.to_numeric(item[spec['target']],errors='coerce')
            if pd.isna(y) or not np.isfinite(y): counts[(name,'missing_target')]+=1;continue
            mol=Chem.MolFromSmiles(smi)
            if mol is None:counts[(name,'invalid_smiles')]+=1;continue
            canon=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
            if canon in seen[name]: counts[(name,'duplicate_molecule')]+=1;continue
            seen[name].add(canon)
            rows.append({'benchmark':name,'smiles':smi,'canonical_smiles':canon,'target':float(y)})
    if not rows: raise ValueError('No valid benchmark molecules!')
    # A canonical structure repeated across different benchmark datasets gets one model embedding.
    unique={}
    for r in rows:unique.setdefault(r['canonical_smiles'],r['smiles'])
    records=[]; indexed={}
    for j,(canon,smi) in enumerate(unique.items()):
        mol=Chem.MolFromSmiles(smi)
        mh=Chem.AddHs(mol)
        if mh.GetNumAtoms()>max_atoms:
            counts[('global','too_many_atoms')]+=1;continue
        pars=AllChem.ETKDGv3();pars.randomSeed=20261112; pars.numThreads=1
        try:
            status=AllChem.EmbedMolecule(mh,pars)
            if status<0: counts[('global','embed_failed')]+=1;continue
            xyz=mh.GetConformer().GetPositions().astype('float32')
            if not np.isfinite(xyz).all():counts[('global','coords_not_finite')]+=1;continue
        except Exception:counts[('global','embed_exception')]+=1;continue
        atoms=np.array([a.GetSymbol() for a in mh.GetAtoms()])
        ind=len(records)
        records.append({'smi':smi,'atoms':atoms,'coordinates':[xyz]})
        indexed[canon]=ind
        if len(records)%500==0: print('ETKDG molecules:',len(records),'/',len(unique),flush=True)
    if not records:raise ValueError('No 3D molecular records generated')
    tab=pd.DataFrame([dict(r,external_lmdb_index=indexed.get(r['canonical_smiles'],-1)) for r in rows])
    tab['has_geometry']=tab['external_lmdb_index']>=0
    # Model data path also needs the exact training dictionary, copied read-only from source.
    import shutil
    # Caller places dictionary into out_dir after path selection; cannot invent a vocabulary.
    map_size=max(2_000_000_000,len(records)*100_000)
    env=lmdb.open(str(lmdb_path),subdir=False,map_size=map_size)
    try:
        with env.begin(write=True) as txn:
            for i,rec in enumerate(records): txn.put(str(i).encode('ascii'),pickle.dumps(rec,protocol=pickle.HIGHEST_PROTOCOL))
        env.sync()
    finally:env.close()
    tab.to_csv(manifest_path,index=False)
    save_json(out_dir/'geometry_build_report.json',{'benchmarks':{k:int((tab.benchmark==k).sum()) for k in BENCHMARKS},
       'unique_structures':len(unique),'generated_3d':len(records),'skipped':{str(k):int(v) for k,v in counts.items()},
       'geometry':'RDKit ETKDGv3 one conformer per structure, fixed seed 20261112, no label used',
       'record_schema':'numeric ASCII key -> pickle dict(smi,atoms,coordinates list)',
       'data_sources':{k:v['url'] for k,v in BENCHMARKS.items()}})
    print('Created read-only-input external LMDB:',lmdb_path,'rows',len(records))
    return tab

def generate_external_embeddings(repo,unicore_dir,checkpoint,external_data_dir,out_dir,batch_size=8,device='cuda:0'):
    """Use *original* UniMol task and encoder CLS in eval mode, no parameter gradients.

    The model is loaded with full checkpoint args and strict state dict. If inference fails,
    outputs remain marked incomplete (never create successful summary prematurely).
    """
    import torch,sys,shutil
    from argparse import Namespace
    repo,unicore_dir,checkpoint,external_data_dir,out_dir=map(Path,(repo,unicore_dir,checkpoint,external_data_dir,out_dir))
    out_dir.mkdir(parents=True,exist_ok=True)
    final_summary=out_dir/'external_embedding_summary.json'
    if final_summary.exists():
        s=json.loads(final_summary.read_text())
        if s.get('status')=='complete':
            print('Reusing existing external embeddings; no model rerun:',out_dir)
            return s
    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise RuntimeError('CUDA not available. Choose heart_gpu GPU Jupyter kernel; no Slurm job is submitted.')
    if not all(p.is_file() for p in [checkpoint,external_data_dir/'valid.lmdb',external_data_dir/'dict.txt']):
        raise FileNotFoundError('Missing checkpoint or external valid.lmdb/dict.txt')
    if str(unicore_dir.resolve()) not in sys.path:sys.path.insert(0,str(unicore_dir.resolve()))
    if str((repo/'model').resolve()) not in sys.path:sys.path.insert(0,str((repo/'model').resolve()))
    import unimol
    from unicore import tasks
    if not (repo/'model'/'unimol').resolve() in Path(unimol.__file__).resolve().parents:
        raise RuntimeError('Wrong unimol Python package; please restart kernel and use original v2 repo')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False) # trusted local model only
    cfg=cp['args'];cfg=Namespace(**cfg) if isinstance(cfg,dict) else cfg
    cfg.data=str(external_data_dir.resolve());cfg.mode='train' # identical to original 10K representation forward
    if hasattr(cfg,'num_workers'):cfg.num_workers=0
    task=tasks.setup_task(cfg)
    model=task.build_model(cfg)
    model.load_state_dict(cp['model'],strict=True)
    model=model.to(torch.device(device)).eval()
    del cp
    task.load_dataset('valid',epoch=1)
    data=task.dataset('valid')
    if hasattr(data,'set_epoch'):data.set_epoch(1)
    tab=pd.read_csv(external_data_dir/'external_manifest.csv')
    n=int(tab.external_lmdb_index.max())+1
    if n!=len(data): raise ValueError(f'Numeric LMDB count mismatch: {n} vs {len(data)}')
    d=int(getattr(cfg,'encoder_embed_dim',512))
    output=out_dir/'external_cls.npy'
    temporary=out_dir/'external_cls.npy.part'
    from numpy.lib.format import open_memmap
    emb=open_memmap(temporary,mode='w+',dtype='float32',shape=(n,d))
    def to_device(v):
        if isinstance(v,torch.Tensor):return v.to(device)
        if isinstance(v,dict):return {k:to_device(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return type(v)(to_device(x) for x in v)
        return v
    buffer={}
    def capture(_module,_inputs,output):
        src=output[0] if isinstance(output,(tuple,list)) else output
        buffer['cls']=src[:,0,:].detach()
    hook=model.encoder.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for start in range(0,n,batch_size):
                stop=min(start+batch_size,n)
                sample=data.collater([data[i] for i in range(start,stop)])
                targets=sample.get('target',{}).get('smi_name',None)
                # Two augmentations: view0 then view1; take view0 to match 10K setup.
                if targets is not None and len(targets)!=2*(stop-start):
                    raise ValueError('Unexpected number of paired views')
                s=to_device(sample)
                buffer.clear()
                model(**s['net_input'],encoder_masked_tokens=s['target']['tokens_target'].ne(task.dictionary.pad()))
                x=buffer['cls']
                if x.shape!=(2*(stop-start),d):raise ValueError('Unexpected CLS shape '+str(x.shape))
                emb[start:stop]=x[:stop-start].float().cpu().numpy()
                if stop%256<batch_size or stop==n:print(f'Frozen HEART external inference: {stop:,}/{n:,}',flush=True)
    finally:
        hook.remove()
    emb.flush();del emb
    temporary.replace(output)
    report={'status':'complete','n_structures':n,'dim':d,'embeddings':str(output),
      'checkpoint':str(checkpoint),'source':'original unimol_contrastive model/encoder CLS view0',
      'inference_mode':'torch.inference_mode, frozen weights, task.mode=train for original augmentation, batch_size='+str(batch_size),
      'geometry':'RDKit ETKDG3D generated, not experimentally determined; inference per structure uses original task preprocessing'}
    save_json(final_summary,report)
    return report


def descriptor_features(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, DataStructs
    n=len(smiles);desc=np.full((n,8),np.nan,np.float32); fp=np.zeros((n,2048),np.float32)
    cans=[];scaffolds=[]
    from rdkit.Chem.Scaffolds import MurckoScaffold
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError('Invalid SMILES after preprocessing '+repr(s))
        desc[i]=[
            Descriptors.HeavyAtomCount(mol),Descriptors.MolWt(mol),Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol)]
        f=AllChem.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(mol)
        ar=np.zeros((2048,),dtype=np.int8);DataStructs.ConvertToNumpyArray(f,ar);fp[i]=ar
        cans.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=True)
        scaffolds.append(scaffold if scaffold else 'ACYCLIC:'+cans[-1])
    return desc,fp,np.array(scaffolds),cans

def overlap_exact_pretrain(manifest,train_unique_sorted,valid_unique_sorted,work_dir):
    """Raw exact-SMILES BLAKE2b-128, same as previous original full audit.
    Does not check canonical equivalence; must report the limitation.
    """
    import subprocess
    work_dir=Path(work_dir);work_dir.mkdir(parents=True,exist_ok=True)
    smi=set(manifest['smiles'].astype(str))
    hashes={hashlib.blake2b(x.encode('utf8'),digest_size=16).hexdigest() for x in smi}
    query=work_dir/'external_raw_hashes_sorted.txt'
    query.write_text(''.join(x+'\n' for x in sorted(hashes)))
    matched=set()
    checked=[]
    for p in (Path(train_unique_sorted),Path(valid_unique_sorted)):
        if not p.exists():continue
        # Pretrain file created by audit_lmdb.py uses C-locale ASCII sort.
        run=subprocess.run(['comm','-12',str(query),str(p)],env={**os.environ,'LC_ALL':'C'},
                           text=True,capture_output=True,check=True)
        matched.update(run.stdout.splitlines());checked.append(str(p))
    hit=manifest['smiles'].map(lambda s:hashlib.blake2b(str(s).encode('utf8'),digest_size=16).hexdigest() in matched)
    return hit,checked


def scaffold_indices(scaffolds,seed=17):
    """Reproducible disjoint-scaffold split, roughly 80/10/10 by molecules.

    Allocate test and valid groups first so neither can accidentally be empty.
    Exact proportions can differ when scaffold groups have very different sizes.
    """
    groups=defaultdict(list)
    for i,sc in enumerate(scaffolds):groups[str(sc)].append(i)
    if len(groups)<3:
        raise ValueError('Need >=3 distinct Murcko scaffold groups for scaffold held-out test')
    rng=np.random.default_rng(seed)
    keys=list(groups);rng.shuffle(keys)
    total=len(scaffolds);desired=max(15,round(total*.10))
    split={'train':[],'valid':[],'test':[]}
    group_count={'train':0,'valid':0,'test':0}
    # Reserve groups for test and validation; remaining groups become training.
    for target in ('test','valid'):
        while len(split[target])<desired and len(keys)>1:
            k=keys.pop();split[target].extend(groups[k]);group_count[target]+=1
    for k in keys:split['train'].extend(groups[k]);group_count['train']+=1
    return {k:np.array(sorted(v),dtype=int) for k,v in split.items()}

def plot_save(fig,path):
    from matplotlib import pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def external_property_evidence(data_dir,embedding_dir,results_dir,audit_dir=None,seeds=(17,31,53),fractions=(0.2,1.0)):
    """Scaffold grouped 80:10:10; train/tune Ridge; test strictly untouched.
    All models share same split. The only trained objects are small linear probes.
    No leakage of experimental labels to HEART pretrained model.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    import matplotlib.pyplot as plt
    data_dir,embedding_dir,results_dir=map(Path,(data_dir,embedding_dir,results_dir))
    results_dir.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(data_dir/'external_manifest.csv')
    cls=np.load(embedding_dir/'external_cls.npy',mmap_mode='r')
    source=source[source.external_lmdb_index>=0].reset_index(drop=True)
    features=np.asarray(cls[source.external_lmdb_index.to_numpy(dtype=int)],dtype='float32')
    if not np.isfinite(features).all():raise ValueError('External CLS contains NaN/Inf')
    summary=[];prediction_rows=[];overlap_info={}
    for bench in BENCHMARKS:
        sub=source[source.benchmark==bench].copy().reset_index(drop=True)
        if len(sub)<60:print('Skipping benchmark with <60 structures',bench);continue
        y=sub['target'].to_numpy(dtype=float)
        desc,fp,scaffolds,cans=descriptor_features(sub.smiles.tolist())
        cls_b=features[source.benchmark==bench]
        # Never delete exact raw overlaps silently. Report and separately flag them.
        overlaps=pd.Series(False,index=sub.index);checked=[]
        if audit_dir is not None:
            a=Path(audit_dir)
            overlaps,checked=overlap_exact_pretrain(sub,a/'train_raw_unique_sorted.txt',a/'valid_raw_unique_sorted.txt',results_dir/bench)
        overlap_info[bench]={'n':len(sub),'raw_pretrain_exact_overlap':int(overlaps.sum()),
            'raw_pretrain_overlap_checked_files':checked,'warning':'Raw exact SMILES only; canonical matches not checked.'}
        if checked:print(bench,'raw exact overlap with pretrain:',int(overlaps.sum()),'of',len(sub))
        else:print(bench,'pretraining-overlap files unavailable: NOT verified')
        matrices={'Mean label':None,'Heavy atoms only':desc[:,[0]],
                  'RDKit descriptors':desc,'Morgan-ECFP4':fp,'Frozen HEART CLS':cls_b}
        for seed in seeds:
            split=scaffold_indices(scaffolds,seed)
            tr,val,te=split['train'],split['valid'],split['test']
            if min(map(len,(tr,val,te)))<15:
                warnings.warn(f'Insufficient scaffold subsets for {bench},seed {seed}: {[len(x) for x in (tr,val,te)]}')
                continue
            group_sets=[set(scaffolds[idx]) for idx in (tr,val,te)]
            assert not any(group_sets[i]&group_sets[j] for i,j in ((0,1),(0,2),(1,2)))
            for fraction in fractions:
                n_train=max(10,round(len(tr)*fraction))
                # fixed, seeded subset of only the TRAIN partition; no test visibility.
                rng=np.random.default_rng(seed+int(1000*fraction))
                train_sel=np.sort(rng.choice(tr,size=min(n_train,len(tr)),replace=False))
                for model_name,X in matrices.items():
                    best_alpha=None
                    if X is None:
                        mdl=DummyRegressor(strategy='mean').fit(np.ones((len(train_sel),1)),y[train_sel]);
                        pred=mdl.predict(np.ones((len(te),1))); best_alpha=np.nan
                    else:
                        # Hyperparameters chosen with validation, never test; no external heavy AutoML.
                        best_mse=np.inf
                        for alpha in (0.1,10.0,1000.0):
                            pipe=make_pipeline(StandardScaler(with_mean=False if model_name=='Morgan-ECFP4' else True),
                                               Ridge(alpha=alpha,solver='lsqr'))
                            pipe.fit(X[train_sel],y[train_sel])
                            mse=mean_squared_error(y[val],pipe.predict(X[val]))
                            if mse<best_mse:best_mse=mse;best_alpha=alpha
                        train_valid=np.concatenate([train_sel,val])
                        mdl=make_pipeline(StandardScaler(with_mean=False if model_name=='Morgan-ECFP4' else True),
                                          Ridge(alpha=best_alpha,solver='lsqr'))
                        mdl.fit(X[train_valid],y[train_valid]);pred=mdl.predict(X[te])
                    err=mean_squared_error(y[te],pred)**.5
                    summary.append({'benchmark':bench,'seed':seed,'train_fraction':fraction,'model':model_name,
                        'n_train':len(train_sel),'n_valid':len(val),'n_test':len(te),
                        'rmse':err,'mae':mean_absolute_error(y[te],pred),
                        'r2':r2_score(y[te],pred),'chosen_alpha':best_alpha,
                        'n_test_raw_pretrain_overlap':int(overlaps.iloc[te].sum()),
                        'raw_overlap_checked':bool(checked)})
                    if fraction==1.0 and seed==seeds[0]:
                        for i,k in enumerate(te):prediction_rows.append({'benchmark':bench,'model':model_name,
                             'smiles':sub.smiles.iloc[k],'actual':y[k],'predicted':pred[i],
                             'seed':seed,'raw_pretrain_overlap':bool(overlaps.iloc[k])})
    if not summary:raise RuntimeError('No benchmark could be evaluated')
    stats=pd.DataFrame(summary);stats.to_csv(results_dir/'benchmark_all_runs.csv',index=False)
    preds=pd.DataFrame(prediction_rows);preds.to_csv(results_dir/'benchmark_test_predictions.csv',index=False)
    save_json(results_dir/'benchmark_overlap_report.json',overlap_info)
    for bench in stats.benchmark.unique():
        d=stats[(stats.benchmark==bench)&(stats.train_fraction==1.0)]
        agg=d.groupby('model',as_index=False).agg(rmse_mean=('rmse','mean'),rmse_std=('rmse','std'))
        agg=agg.sort_values('rmse_mean')
        fig,ax=plt.subplots(figsize=(8,4.6))
        ax.barh(agg.model,agg.rmse_mean,xerr=agg.rmse_std.fillna(0),capsize=3)
        ax.invert_yaxis();ax.set_xlabel('Held-out scaffold RMSE (lower is better)')
        ax.set_title(f'{bench}: frozen HEART vs baselines ({len(seeds)} scaffold splits)')
        fig.tight_layout();plot_save(fig,results_dir/f'{bench}_benchmark_RMSE.png')
        d=preds[(preds.benchmark==bench)&(preds.model=='Frozen HEART CLS')]
        if len(d):
            fig,ax=plt.subplots(figsize=(5,4.6))
            ax.scatter(d.actual,d.predicted,s=13,alpha=.65)
            lo=min(d.actual.min(),d.predicted.min());hi=max(d.actual.max(),d.predicted.max())
            ax.plot([lo,hi],[lo,hi],linestyle='--',color='gray')
            ax.set_xlabel('Experimental');ax.set_ylabel('HEART linear probe prediction')
            ax.set_title(f'{bench} — unseen scaffold test (seed {seeds[0]})')
            fig.tight_layout();plot_save(fig,results_dir/f'{bench}_HEART_pred_vs_experiment.png')
    # Sample-efficiency figure (20% vs 100% labeled training records)
    for bench in stats.benchmark.unique():
        d=stats[(stats.benchmark==bench)&(stats.model.isin(['Frozen HEART CLS','Morgan-ECFP4','RDKit descriptors']))]
        g=d.groupby(['model','train_fraction'],as_index=False).agg(mean=('rmse','mean'),std=('rmse','std'))
        fig,ax=plt.subplots(figsize=(7,4))
        for model in g.model.unique():
            gg=g[g.model==model].sort_values('train_fraction')
            ax.errorbar(gg.train_fraction*100,gg['mean'],yerr=gg['std'].fillna(0),marker='o',capsize=3,label=model)
        ax.set_xlabel('Fraction of labeled training set (%)');ax.set_ylabel('Scaffold test RMSE')
        ax.set_title(f'{bench}: labeled-data efficiency');ax.legend(fontsize=8)
        fig.tight_layout();plot_save(fig,results_dir/f'{bench}_label_efficiency.png')
    print(stats.groupby(['benchmark','model'])['rmse'].agg(['mean','std']).round(3).to_string())
    return stats


def similarity_evidence(old_run,manifest_dir,out_dir,n_anchor=400,n_gallery=4000,seed=17):
    """Structural nearest-neighbor retrieval, held at matched heavy-atom counts.

    View0 CLS embeddings and numeric-key manifest from EXISTING 10K run. No GPU.
    Positive result is NOT an independent experimental property benchmark.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem,Descriptors,Draw
    from scipy.stats import spearmanr,wilcoxon
    import matplotlib.pyplot as plt
    RDLogger.DisableLog('rdApp.warning')
    old_run,manifest_dir,out_dir=map(Path,(old_run,manifest_dir,out_dir));out_dir.mkdir(parents=True,exist_ok=True)
    emb=np.load(old_run/'cls_view0.npy',mmap_mode='r')
    meta=pd.read_csv(manifest_dir/'sample_metadata.csv',keep_default_na=False)
    if len(emb)!=len(meta):raise ValueError('10K metadata and CLS order mismatch')
    rng=np.random.default_rng(seed)
    valid=[]
    for k,s in enumerate(meta.smiles.astype(str)):
        m=Chem.MolFromSmiles(s)
        if m is not None:valid.append((k,m,Descriptors.HeavyAtomCount(m)))
    if len(valid)<n_gallery:raise ValueError('Not enough valid molecules for gallery')
    gallery_idx=rng.choice(len(valid),size=n_gallery,replace=False)
    ga=np.array([valid[j][0] for j in gallery_idx]);mols=[valid[j][1] for j in gallery_idx]
    heavy=np.array([valid[j][2] for j in gallery_idx]);
    from rdkit.Chem import rdFingerprintGenerator
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[gen.GetFingerprint(m) for m in mols]
    anchor_pos=rng.choice(n_gallery,size=min(n_anchor,n_gallery),replace=False)
    vectors=np.asarray(emb[ga],dtype='float32');vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)+1e-8
    cos=vectors[anchor_pos] @ vectors.T
    for j,idx in enumerate(anchor_pos): cos[j,idx]=-np.inf
    # Mask duplicate chemical structures to avoid trivial exact-match retrieval.
    canon=[Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) for m in mols]
    bycanon=defaultdict(list)
    for j,s in enumerate(canon):bycanon[s].append(j)
    for i,anchor in enumerate(anchor_pos):cos[i,bycanon[canon[anchor]]]=-np.inf
    nn=cos.argmax(axis=1)
    rows=[]
    for j,a in enumerate(anchor_pos):
        cand=np.flatnonzero((np.abs(heavy-heavy[a])<=2)&(np.arange(n_gallery)!=a))
        cand=np.array([k for k in cand if canon[k]!=canon[a]])
        if not len(cand):continue
        ctrl=int(rng.choice(cand))
        near=int(nn[j])
        a_fp=fps[a]
        rows.append({'anchor_index':int(ga[a]),'nearest_index':int(ga[near]),'size_control_index':int(ga[ctrl]),
            'near_tanimoto':float(DataStructs.TanimotoSimilarity(a_fp,fps[near])),
            'control_tanimoto':float(DataStructs.TanimotoSimilarity(a_fp,fps[ctrl])),
            'near_heavy_atom_delta':int(abs(heavy[a]-heavy[near])),
            'control_heavy_atom_delta':int(abs(heavy[a]-heavy[ctrl])),
            'near_cosine':float(cos[j,near]),'anchor_smiles':meta.smiles.iloc[ga[a]],
            'nearest_smiles':meta.smiles.iloc[ga[near]],'control_smiles':meta.smiles.iloc[ga[ctrl]]})
    tab=pd.DataFrame(rows);tab.to_csv(out_dir/'structural_retrieval_pairs.csv',index=False)
    delta=tab.near_tanimoto-tab.control_tanimoto
    boot=[]
    for _ in range(1000):boot.append(float(np.mean(rng.choice(delta,size=len(delta),replace=True))))
    result={'n_anchor':len(tab),'gallery_size':n_gallery,'mean_near_tanimoto':tab.near_tanimoto.mean(),
        'mean_size_matched_random_tanimoto':tab.control_tanimoto.mean(),
        'mean_enrichment':float(delta.mean()),'bootstrap_95ci_mean_enrichment':list(np.quantile(boot,[.025,.975])),
        'paired_wilcoxon_pvalue':float(wilcoxon(delta).pvalue),
        'note':'Nearest neighbor in 4k CLS gallery (same source validation set); structural reference ECFP4 Tanimoto; controls within +/-2 heavy atoms. This is structure sensitivity, not experimental downstream transfer.'}
    save_json(out_dir/'structural_retrieval_summary.json',result)
    fig,ax=plt.subplots(figsize=(5.5,4.5))
    ax.scatter(tab.control_tanimoto,tab.near_tanimoto,s=10,alpha=.4)
    ax.plot([0,1],[0,1],ls='--',color='gray')
    ax.set_xlabel('Size-matched random: Tanimoto');ax.set_ylabel('HEART nearest: Tanimoto')
    ax.set_title('Does CLS proximity reflect molecular topology?')
    fig.tight_layout();plot_save(fig,out_dir/'CLS_structure_retrieval.png')
    # Show representative molecular triplets (direct chemical visual, not generated illustration).
    best=tab.sort_values('near_tanimoto',ascending=False).iloc[:3]
    grid=[];legends=[]
    for _,r in best.iterrows():
        for role,col in [('Anchor','anchor_smiles'),('HEART nearest','nearest_smiles'),('Size control','control_smiles')]:
            grid.append(Chem.MolFromSmiles(r[col]));legends.append(role)
    pic=Draw.MolsToGridImage(grid,molsPerRow=3,subImgSize=(280,190),legends=legends,returnPNG=True)
    (out_dir/'Example_neighbor_molecules.png')
    print('Structure retrieval:',json.dumps(result,indent=2))
    return tab


def existing_10k_slide_figures(old_root,out_dir):
    """Create clean, slide-ready figures using existing 10K summaries/CSV only."""
    import matplotlib.pyplot as plt
    old_root,out_dir=Path(old_root),Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    runs={k:old_root/f'{k}_10k_batch8' for k in ('train','full')}
    for k,p in runs.items():
        if not (p/'evaluation_summary.json').exists():
            raise FileNotFoundError('Existing 10K summary missing: '+str(p))
    summary={k:json.loads((p/'evaluation_summary.json').read_text()) for k,p in runs.items()}
    mapping={'Train':'train','Validation':'full'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    x=np.arange(2)
    a=[summary[k]['masked_atom_accuracy']*100 for k in ('train','full')]
    base=[summary[k].get('masked_atom_majority_class_baseline_accuracy',np.nan)*100 for k in ('train','full')]
    axes[0].bar(x-.18,a,.36,label='HEART')
    axes[0].bar(x+.18,base,.36,label='Majority atom baseline')
    axes[0].set_xticks(x,['Train 10K','Validation 10K']);axes[0].set_ylabel('Masked atom accuracy (%)')
    axes[0].legend(fontsize=8);axes[0].set_ylim(0,105)
    noisy=summary['full']['coord_noisy_rmse'];pred=summary['full']['coord_pred_rmse']
    axes[1].bar(['Noisy input','HEART recovered'],[noisy,pred]);axes[1].set_ylabel('Masked coordinate RMSE')
    axes[1].set_title('Validation 10K geometry denoising')
    fig.suptitle('Original HEART H200 pretraining objectives: independent validation molecules')
    fig.tight_layout();plot_save(fig,out_dir/'slide_1_structure_geometry.png')
    atom=pd.read_csv(runs['full']/'atom_token_class_accuracy.csv').head(8)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(atom.token_symbol.astype(str),atom.accuracy*100)
    ax.set_ylim(0,105);ax.set_ylabel('Masked atom accuracy (%)')
    ax.set_title('Validation 10K: atom accuracy by element')
    for i,r in atom.reset_index(drop=True).iterrows():ax.text(i,r.accuracy*100+1,str(int(r.masked_targets)),ha='center',fontsize=7)
    fig.tight_layout();plot_save(fig,out_dir/'slide_1_atom_class_accuracy.png')
    save_json(out_dir/'existing_evidence_used.json',{'train':str(runs['train']),'validation':str(runs['full']),
        'note':'Pretraining objective evidence only, not external property generalization.'})
    return summary


def geometry_example(repo,unicore_dir,checkpoint,ligand_data_dir,manifest_dir,out_dir,n=3,device='cuda:0'):
    """Optional actual model coordinate triptychs from held-out LMDB molecules (3 only).

    Outputs normalized task coordinates; points are atoms and masked atoms, not bond-perfect 3D depictions.
    Never modifies checkpoint/dataset; no Slurm.
    """
    import torch,sys
    import matplotlib.pyplot as plt
    from argparse import Namespace
    repo,unicore_dir,checkpoint,ligand_data_dir,manifest_dir,out_dir=map(Path,(repo,unicore_dir,checkpoint,ligand_data_dir,manifest_dir,out_dir))
    out_dir.mkdir(parents=True,exist_ok=True)
    for p in (unicore_dir,repo/'model'):
        if str(p.resolve()) not in sys.path:sys.path.insert(0,str(p.resolve()))
    from unicore import tasks
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=cp['args'];cfg=Namespace(**cfg) if isinstance(cfg,dict) else cfg
    cfg.data=str(ligand_data_dir);cfg.mode='train'
    task=tasks.setup_task(cfg);model=task.build_model(cfg)
    model.load_state_dict(cp['model'],strict=True);model=model.to(device).eval();del cp
    task.load_dataset('valid',epoch=1);ds=task.dataset('valid')
    if hasattr(ds,'set_epoch'):ds.set_epoch(1)
    tab=pd.read_csv(manifest_dir/'sample_metadata.csv')
    idx=tab.lmdb_index.iloc[:n].astype(int).tolist()
    for j,k in enumerate(idx):
        sample=ds.collater([ds[k]])
        def mv(x):
            if isinstance(x,torch.Tensor):return x.to(device)
            if isinstance(x,dict):return {a:mv(b) for a,b in x.items()}
            return x
        s=mv(sample)
        mask=s['target']['tokens_target'].ne(task.dictionary.pad())
        with torch.inference_mode():outputs=model(**s['net_input'],encoder_masked_tokens=mask)
        predicted=outputs[2][0].detach().cpu().numpy()
        truth=s['target']['coord_target'][0].detach().cpu().numpy()
        noisy=s['net_input']['src_coord'][0].detach().cpu().numpy()
        m=mask[0].detach().cpu().numpy().astype(bool)
        tk=s['net_input']['src_tokens'][0]
        valid=(tk.ne(task.dictionary.pad()) & tk.ne(task.dictionary.bos()) & tk.ne(task.dictionary.eos())).detach().cpu().numpy()
        fig=plt.figure(figsize=(12,3.7))
        all_vals=np.concatenate([truth[valid],noisy[valid],predicted[valid]])
        low=all_vals.min(axis=0);hi=all_vals.max(axis=0);span=float(np.max(hi-low));center=(low+hi)/2
        for i,(data,label) in enumerate([(truth,'Target (normalized)'),(noisy,'Noisy input'),(predicted,'HEART prediction')]):
            ax=fig.add_subplot(1,3,i+1,projection='3d')
            ax.scatter(*data[valid].T,s=9,alpha=.4,color='gray')
            ax.scatter(*data[m].T,s=24,alpha=.85)
            for a in range(3):
                ((ax.set_xlim,ax.set_ylim,ax.set_zlim)[a])(center[a]-span*.55,center[a]+span*.55)
            ax.set_title(label);ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
        fig.suptitle('Original HEART 3D denoising, masked atoms highlighted — validation LMDB index '+str(k))
        fig.tight_layout();plot_save(fig,out_dir/f'geometry_case_{j+1}_index{k}.png')
    return idx

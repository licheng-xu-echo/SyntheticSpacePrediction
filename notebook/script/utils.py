# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:28:42 2022

@author: LiCheng_Xu
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,Descriptors
from ase import Atoms as ASE_Atoms
from rdkit.ML.Descriptors import MoleculeDescriptors
from dscribe.descriptors import MBTR,ACSF,SOAP,LMBTR
from openbabel.pybel import (readfile,Outputfile)

period_table = Chem.GetPeriodicTable()
descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
def box_cox_trans(x,lambda_):
    '''
    Box-Cox Transformation

    Parameters
    ----------
    x : ndarray
        DESCRIPTION.
    lambda_ : float
        DESCRIPTION.

    Returns
    -------
    ndarray
        DESCRIPTION.

    '''
    if lambda_ != 0:
        return (np.power(x,lambda_)-1)/lambda_
    else:
        return np.log(x)

def de_box_cox_trans(x,lambda_):
    
    if lambda_ != 0:
        return np.power((1+lambda_*x),1/lambda_)
    else:
        return np.exp(np.power(x,lambda_))
    
def log_trans(x):
    '''
    Logarithmic Transformation

    Parameters
    ----------
    x : ndarray
        DESCRIPTION.

    Returns
    -------
    ndarray
        DESCRIPTION.

    '''
    
    return np.log((1-x)/(1+x))

def de_log_trans(x):
    return (1-np.exp(x))/(1+np.exp(x))

def ee2ddG(ee,T):
    '''
    Transformation from ee to ΔΔG
    Parameters
    ----------
    ee : ndarray
        Enantiomeric excess.
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ddG : ndarray
        ΔΔG (kcal/mol).
    '''
    
    ddG = np.abs(8.314 * T * np.log((1-ee)/(1+ee)))  # J/mol
    ddG = ddG/1000/4.18            # kcal/mol
    return ddG

def ddG2ee(ddG,T):
    '''
    Transformation from ΔΔG to ee. 
    Parameters
    ----------
    ddG : ndarray
        ΔΔG (kcal/mol).
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ee : ndarray
        Absolute value of enantiomeric excess.
    '''
    
    ddG = ddG*1000*4.18
    ee = (1-np.exp(ddG/(8.314*T)))/(1+np.exp(ddG/(8.314*T)))
    return np.abs(ee)

def maxminscale(array):
    '''
    Max-min scaler

    Parameters
    ----------
    array : ndarray
        Original numpy array.

    Returns
    -------
    array : ndarray
        numpy array with max-min scaled.

    '''
    return (array - array.min(axis=0))/(array.max(axis=0)-array.min(axis=0))

def genDescMap(map_csv,desc_name=False):
    desc_map = pd.read_csv(map_csv,index_col=0)
    desc_names = list(desc_map.columns)
    desc_map = {smi:desc for smi,desc in zip(list(desc_map.index),
                                                   desc_map.to_numpy())}
    
    if desc_name:
        return desc_map,desc_names
    else:
        return desc_map
def Mol2Atoms(mol):
    positions = mol.GetConformer().GetPositions()
    atom_types = [period_table.GetElementSymbol(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    atoms = ASE_Atoms(symbols=atom_types,positions=positions)
    return atoms

def genMolandAtom(smi):
    assert smi != '', "Empty SMILES was provided!"
    mol = AllChem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    atoms = Mol2Atoms(mol)
    return mol,atoms
def readMolandAtom(file):
    with open(file,'r') as fr:
        smi = fr.readlines()[0].strip()
    mol = Chem.MolFromMolFile(file,removeHs=False,sanitize=False)
    atoms = Mol2Atoms(mol)
    mol = AllChem.AddHs(Chem.MolFromSmiles(smi))
    return smi,mol,atoms
def getmorganfp(mol,radius=2,nBits=2048,useChirality=True):
    '''
    
    Parameters
    ----------
    mol : mol
        RDKit mol object.

    Returns
    -------
    mf_desc_map : ndarray
        ndarray of molecular fingerprint descriptors.

    '''
    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius=radius,nBits=nBits,useChirality=useChirality)
    return np.array(list(map(eval,list(fp.ToBitString()))))

def calc_rdkit_desc(mol):
    return desc_calc.CalcDescriptors(mol)

def get_atom_species(smiles_set,smi_mol_map):
    species = []
    for smi in smiles_set:
        species += [atom.GetSymbol() for atom in smi_mol_map[smi].GetAtoms()]
    return list(set(species))

def calc_Dscribe_Desc(smi,smi_atoms_map,species,parameter_dict,type_='MBTR'):
    type_ = type_.lower()
    
    if type_ == 'mbtr':
        k1 = parameter_dict['k1']
        k2 = parameter_dict['k2']
        k3 = parameter_dict['k3']
        periodic = parameter_dict['periodic']
        normalization = parameter_dict['normalization']
        calculator = MBTR(species=species,k1=k1,k2=k2,k3=k3,periodic=periodic,normalization=normalization)
        return calculator.create(smi_atoms_map[smi])
    
    elif type_ == 'acsf':
        rcut = parameter_dict['rcut']
        g2_params = parameter_dict['g2_params']
        g4_params = parameter_dict['g4_params']
        calculator = ACSF(species=species,rcut=rcut,g2_params=g2_params,g4_params=g4_params)
        return np.mean(calculator.create(smi_atoms_map[smi]),axis=0)
    
    elif type_ == 'soap':
        rcut = parameter_dict['rcut']
        nmax = parameter_dict['nmax']
        lmax = parameter_dict['lmax']
        periodic = parameter_dict['periodic']
        calculator = SOAP(species=species,periodic=periodic,rcut=rcut,nmax=nmax,lmax=lmax)
        return np.mean(calculator.create(smi_atoms_map[smi]),axis=0)

    elif type_ == 'lmbtr':
        k2 = parameter_dict['k2']
        k3 = parameter_dict['k3']
        periodic = parameter_dict['periodic']
        normalization = parameter_dict['normalization']
        calculator = LMBTR(species=species,k2=k2,k3=k3,periodic=False,normalization=normalization)
        return np.mean(calculator.create(smi_atoms_map[smi]),axis=0)

def process_desc(array):

    array = np.array(array,dtype=np.float32)
    desc_len = array.shape[1]
    rig_idx = []
    for i in range(desc_len):
        try:
            desc_range = array[:,i].max() - array[:,i].min()
            if desc_range != 0 and not np.isnan(desc_range):
                rig_idx.append(i)
        except:
            continue
    array = array[:,rig_idx]
    return array
def genCompoundWiseDesc(biaryl_smiles,olefin_smiles,tdg_smiles,desc_map,scaler=True):
    if scaler:
        b_desc = maxminscale(process_desc(np.array([desc_map[smi] for smi in biaryl_smiles])))
        o_desc = maxminscale(process_desc(np.array([desc_map[smi] for smi in olefin_smiles])))
        t_desc = maxminscale(process_desc(np.array([desc_map[smi] for smi in tdg_smiles])))
    else:
        b_desc = process_desc(np.array([desc_map[smi] for smi in biaryl_smiles]))
        o_desc = process_desc(np.array([desc_map[smi] for smi in olefin_smiles]))
        t_desc = process_desc(np.array([desc_map[smi] for smi in tdg_smiles]))
    compound_desc = np.concatenate([b_desc,o_desc,t_desc],axis=1)
    return compound_desc

def MolFormatConversion(input_file:str,output_file:str,input_format="xyz",output_format="sdf"):
    molecules = readfile(input_format,input_file)
    output_file_writer = Outputfile(output_format,output_file,overwrite=True)
    for i,molecule in enumerate(molecules):
        output_file_writer.write(molecule)
    output_file_writer.close()
    print('%d molecules converted'%(i+1)) 
    
    
def genDescDataset(physorg_desc,physorg_desc_names,target,target_name='ddG(kcal/mol)'):
    phyorg_desc_dataset = np.concatenate([physorg_desc,target.reshape(-1,1)],axis=1)
    title_of_dataset = list(physorg_desc_names) + [target_name]
    desc_dataset = pd.DataFrame.from_records(phyorg_desc_dataset)
    desc_dataset.columns = title_of_dataset
    return desc_dataset    
    
    
    
    
    
    
    
    
    
    
    



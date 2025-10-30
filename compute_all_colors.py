import numpy
import healpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mwdust
from sklearn.neighbors import KernelDensity


db= "filtered_data_photometry.csv"
db = pd.read_csv(db)
db['e_mag_psf_fix']=((db['e_mag_psf'].astype('float64')**2.0) + (0.01**2.0))**(1.0/2.0)
db['invar']=1.0/(db['e_mag_psf_fix'])**2.0
db['var']=db['e_mag_psf_fix']**2.0
db['ones']=1.0

dbu=db[(db['filter']=='u')]
dbv=db[(db['filter']=='v')]
dbg=db[(db['filter']=='g')]
dbr=db[(db['filter']=='r')]
dbi=db[(db['filter']=='i')]
dbz=db[(db['filter']=='z')]

def wavg(group, avg_name, invar):
    d = group[avg_name]
    w = group[invar]
    return (d * w).sum() / w.sum()

def propvar(group, var, ones):
    v = group[var]
    w = group[ones] #sorrynotsorry
    return v.sum()/((w.sum())**2.0)

print("Starting uvg computation")
uvar_nightly = dbu.groupby(['object_id', 'night_mjd']).apply(propvar, 'var', 'ones')
umag_nightly = dbu.groupby(['object_id', 'night_mjd']).apply(wavg, 'mag_psf', 'invar')
umag_nightly=umag_nightly.reset_index()
uvar_nightly=uvar_nightly.reset_index()
print("Merging umag and uvar")
u_nightly=umag_nightly.merge(uvar_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('mag', 'var'))
u_nightly['0invar']=1/u_nightly['0var']

print("Computing vvar and vmag")
vvar_nightly = dbv.groupby(['object_id', 'night_mjd']).apply(propvar, 'var', 'ones')
vmag_nightly = dbv.groupby(['object_id', 'night_mjd']).apply(wavg, 'mag_psf', 'invar')
vmag_nightly=vmag_nightly.reset_index()
vvar_nightly=vvar_nightly.reset_index()
print("Merging vmag and vvar")
v_nightly=vmag_nightly.merge(vvar_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('mag', 'var'))
v_nightly['0invar']=1/v_nightly['0var']

print("Computing gvar and gmag")
gvar_nightly = dbg.groupby(['object_id', 'night_mjd']).apply(propvar, 'var', 'ones')
gmag_nightly = dbg.groupby(['object_id', 'night_mjd']).apply(wavg, 'mag_psf', 'invar')
gmag_nightly=gmag_nightly.reset_index()
gvar_nightly=gvar_nightly.reset_index()
g_nightly=gmag_nightly.merge(gvar_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('mag', 'var'))
g_nightly['0invar']=1/g_nightly['0var']

print("Computing rvar and rmag")
rvar_nightly = dbr.groupby(['object_id', 'night_mjd']).apply(propvar, 'var', 'ones')
rmag_nightly = dbr.groupby(['object_id', 'night_mjd']).apply(wavg, 'mag_psf', 'invar')
rmag_nightly=rmag_nightly.reset_index()
rvar_nightly=rvar_nightly.reset_index()
r_nightly=rmag_nightly.merge(rvar_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('mag', 'var'))
r_nightly['0invar']=1/r_nightly['0var']

print("Computing ivar and imag")
ivar_nightly = dbi.groupby(['object_id', 'night_mjd']).apply(propvar, 'var', 'ones')
imag_nightly = dbi.groupby(['object_id', 'night_mjd']).apply(wavg, 'mag_psf', 'invar')
imag_nightly=imag_nightly.reset_index()
ivar_nightly=ivar_nightly.reset_index()
i_nightly=imag_nightly.merge(ivar_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('mag', 'var'))
i_nightly['0invar']=1/i_nightly['0var']

print("Computing zvar and zmag")
zvar_nightly = dbz.groupby(['object_id', 'night_mjd']).apply(propvar, 'var', 'ones')
zmag_nightly = dbz.groupby(['object_id', 'night_mjd']).apply(wavg, 'mag_psf', 'invar')
zmag_nightly=zmag_nightly.reset_index()
zvar_nightly=zvar_nightly.reset_index()
z_nightly=zmag_nightly.merge(zvar_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('mag', 'var'))
z_nightly['0invar']=1/z_nightly['0var']

print("Computing ugvar and ug")
ug_nightly = u_nightly.merge(g_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('_u', '_g'))
ug_nightly['ug']=ug_nightly['0mag_u']-ug_nightly['0mag_g']
ug_nightly['ugvar']=ug_nightly['0var_u']+ug_nightly['0var_g']
ug_nightly['uginvar']=1.0/ug_nightly['ugvar']
ug = ug_nightly.groupby('object_id').apply(wavg, 'ug', 'uginvar')
ug = ug.reset_index()

print("Computing uvvar and uv")
uv_nightly = u_nightly.merge(v_nightly, how='inner', on=['object_id', 'night_mjd'], suffixes=('_u', '_v'))
uv_nightly['uv']=uv_nightly['0mag_u']-uv_nightly['0mag_v']
uv_nightly['uvvar']=uv_nightly['0var_u']+uv_nightly['0var_v']
uv_nightly['uvinvar']=1.0/uv_nightly['uvvar']
uv = uv_nightly.groupby('object_id').apply(wavg, 'uv', 'uvinvar')
uv = uv.reset_index()

print("Merging ug and uv")
uvg = ug.merge(uv, how='inner', on='object_id', suffixes=('ug', 'uv'))

print("Length of uvg: ", len(uvg))
print("Columns of uvg: ", uvg.columns)


uvg.to_csv('adrian_weighted_colors_w3w4.csv', index=False)

import os
from astroquery.utils.tap.core import TapPlus
from astropy.table import Table
import pyvo as vo
import time
from astropy.table import vstack


#tap_service = vo.dal.TAPService("https://api.skymapper.nci.org.au/public/tap")
tap = TapPlus(url="https://api.skymapper.nci.org.au/public/tap/")


#ex_query = "SELECT m.object_id, g.source_id, m.raj2000,  m.dej2000, m.glon, m.glat, m.u_ngood, m.u_nclip, m.v_ngood, m.g_ngood, m.r_ngood, m.i_ngood, m.z_ngood, w.w1mpro, w.w1sigmpro, w.w2mpro, w.w2sigmpro, t.j_m, t.j_msigcom, t.h_m, t.h_msigcom, t.k_m, t.k_msigcom, m.u_psf, m.e_u_psf, m.v_psf, m.e_v_psf, m.g_psf, m.e_g_psf, m.r_psf, m.e_r_psf, m.i_psf, m.e_i_psf, m.z_psf, m.e_z_psf, m.self_dist1, t.prox AS tprox, m.allwise_dist, g.parallax, g.parallax_error, g.astrometric_excess_noise, g.astrometric_excess_noise_sig, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, m.ebmv_sfd FROM dr4.master m JOIN ext.twomass_psc t ON m.twomass_key=t.pts_key JOIN ext.gaia_dr3 g ON m.gaia_dr3_id1=g.source_id JOIN ext.allwise w ON m.allwise_cntr=w.cntr WHERE m.twomass_dist < 2.0 AND m.gaia_dr3_dist1 < 2.0  AND m.allwise_dist < 3.0 AND m.self_dist1 > 6.0 AND t.ph_qual = 'AAA' AND t.gal_contam = 0 AND t.ext_key IS NULL AND t.cc_flg = '000'  AND m.class_star > 0.9 AND flags_psf = 0 AND m.u_ngood > 0 AND m.v_ngood > 0 AND m.g_ngood > 0 AND (t.j_m - t.k_m) > 0.85  AND g.parallax>0 AND g.parallax <100  AND (t.j_m - 5 * log(100.0/g.parallax)) < 14.0 AND (t.j_m - t.k_m - 0.413*(m.ebmv_sfd))>0.85 "
#ex_query = "SELECT m.object_id, g.source_id, m.raj2000,  m.dej2000, m.glon, m.glat, m.u_ngood, m.u_nclip, m.v_ngood, m.g_ngood, m.r_ngood, m.i_ngood, m.z_ngood, w.w1mpro, w.w1sigmpro, w.w2mpro, w.w2sigmpro, t.j_m, t.j_msigcom, t.h_m, t.h_msigcom, t.k_m, t.k_msigcom, m.u_psf, m.e_u_psf, m.v_psf, m.e_v_psf, m.g_psf, m.e_g_psf, m.r_psf, m.e_r_psf, m.i_psf, m.e_i_psf, m.z_psf, m.e_z_psf, m.prox, t.prox AS tprox, m.allwise_dist, g.parallax, g.parallax_error, g.astrometric_excess_noise, g.astrometric_excess_noise_sig, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, m.ebmv_sfd FROM dr2.master m JOIN ext.twomass_psc t ON m.twomass_key=t.pts_key JOIN ext.gaia_dr2 g ON m.gaia_dr2_id1=g.source_id JOIN ext.allwise w ON m.allwise_cntr=w.cntr WHERE m.twomass_dist < 2.0 AND m.gaia_dr2_dist1 < 2.0  AND m.allwise_dist < 3.0 AND m.prox > 6.0 AND t.ph_qual = 'AAA' AND t.gal_contam = 0 AND t.ext_key IS NULL AND t.cc_flg = '000'  AND m.class_star > 0.9 AND flags_psf = 0 AND m.u_ngood > 0 AND m.v_ngood > 0 AND m.g_ngood > 0 AND (t.j_m - t.k_m) > 0.85  AND g.parallax>0 AND g.parallax <100  AND (t.j_m - 5 * log(100.0/g.parallax)) < 14.0 "
# ex_query = "SELECT m.object_id, g.source_id, m.raj2000,  m.dej2000, m.glon, m.glat, m.u_ngood, m.u_nclip, m.v_ngood, m.g_ngood, m.r_ngood, m.i_ngood, m.z_ngood, w.w1mpro, w.w1sigmpro, w.w2mpro, w.w2sigmpro, t.j_m, t.j_msigcom, t.h_m, t.h_msigcom, t.k_m, t.k_msigcom, m.u_psf, m.e_u_psf, m.v_psf, m.e_v_psf, m.g_psf, m.e_g_psf, m.r_psf, m.e_r_psf, m.i_psf, m.e_i_psf, m.z_psf, m.e_z_psf, m.prox, t.prox AS tprox, m.allwise_dist, g.parallax, g.parallax_error, g.astrometric_excess_noise, g.astrometric_excess_noise_sig, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, m.ebmv_sfd FROM dr2.master m JOIN ext.twomass_psc t ON m.twomass_key=t.pts_key JOIN ext.gaia_dr2 g ON m.gaia_dr2_id1=g.source_id JOIN ext.allwise w ON m.allwise_cntr=w.cntr WHERE m.twomass_dist < 2.0 AND m.gaia_dr2_dist1 < 2.0  AND m.allwise_dist < 3.0 AND m.nch_max = 1 AND m.prox > 6.0 AND t.ph_qual = 'AAA' AND t.gal_contam = 0 AND t.ext_key IS NULL AND t.cc_flg = '000'  AND m.class_star > 0.9 AND flags_psf = 0 AND m.u_ngood > 0 AND m.v_ngood > 0 AND m.g_ngood > 0 AND (t.j_m - t.k_m) > 0.85 AND t.j_m  < 14.0 "
# ... existing code ...

ex_query = """
SELECT 
    m.object_id, g.source_id, m.raj2000, m.dej2000, m.glon, 
    m.glat, m.u_ngood, m.u_nclip, m.v_ngood, m.g_ngood, 
    m.r_ngood, m.i_ngood, m.z_ngood, w.w1mpro, w.w1sigmpro, 
    w.w2mpro, w.w2sigmpro, w.w3mpro, w.w4mpro,t.j_m, t.j_msigcom, t.h_m, 
    t.h_msigcom, t.k_m, t.k_msigcom, m.u_psf, m.e_u_psf, m.v_psf, 
    m.e_v_psf, m.g_psf, m.e_g_psf, m.r_psf, m.e_r_psf, m.i_psf, 
    m.e_i_psf, m.z_psf, m.e_z_psf, m.prox, t.prox AS tprox, 
    m.allwise_dist, g.parallax, g.parallax_error, 
    g.astrometric_excess_noise, g.astrometric_excess_noise_sig, 
    g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, m.ebmv_sfd 
FROM dr2.master m 
JOIN ext.twomass_psc t ON m.twomass_key=t.pts_key 
JOIN ext.gaia_dr2 g ON m.gaia_dr2_id1=g.source_id 
JOIN ext.allwise w ON m.allwise_cntr=w.cntr 
WHERE m.twomass_dist < 2.0 
    AND m.gaia_dr2_dist1 < 2.0 
    AND m.allwise_dist < 3.0 
    AND m.prox > 6.0 
    AND t.ph_qual = 'AAA' 
    AND t.gal_contam = 0 
    AND t.ext_key IS NULL 
    AND t.cc_flg = '000' 
    AND m.class_star > 0.9 
    AND flags_psf = 0 
    AND m.u_ngood > 0 
    AND m.v_ngood > 0 
    AND m.g_ngood > 0 
    AND m.nch_max = 1 
    AND (t.j_m - t.k_m) > 0.85 
    AND t.j_m < 14.0
"""

# ... rest of the code ...
#result = tap_service.search(ex_query)

#print(result)
# job = tap_service.submit_job(ex_query)
# job.run_job_async()
# while job.phase != 'COMPLETED':
#      time.sleep(5)  # Check every 5 seconds
#      job.phase
# result = job.fetch_result()

# table=result.to_table()
# table.write ('Dr2_table.csv',format='csv',overwrite=True)
# print("Number of rows: ", len(table))


def execute_tap_query(query):
    """Execute query using TapPlus with status monitoring"""
    try:
        # Launch the job
        print("Submitting job...")
        job = tap.launch_job_async(query)
        
        # Monitor the job status
        while job.get_phase() not in ['COMPLETED', 'ERROR', 'ABORTED']:
            print(f"Job status: {job.get_phase()}")
            time.sleep(10)  # Check status every 10 seconds
            
        if job.get_phase() == 'ERROR':
            raise Exception(f"Job failed: {job.get_error()}")
        elif job.get_phase() == 'ABORTED':
            raise Exception("Job was aborted")
            
        # Get the results
        return job.get_results()
        
    except Exception as e:
        print(f"Query failed: {str(e)}")
        raise

try:
    # Execute the query
    print("Starting query execution...")
    table = execute_tap_query(ex_query)
    
    # Convert to table and save
    #table = result.to_table()
    table.write('Dr2_table_w3w4.csv', format='csv', overwrite=True)
    print("\nFinal Results:")
    print("Total number of rows:", len(table))
    
except Exception as e:
    print(f"Error during execution: {str(e)}")
    # If we have partial results, save them
    if 'table' in locals() and table is not None:
        #table = result.to_table()
        error_file = 'Dr2_table_w3w4_error.csv'
        table.write(error_file, format='csv', overwrite=True)
        print(f"Partial results saved to {error_file}")
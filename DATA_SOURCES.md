# External data and extended analyses

The code snapshot omits third-party raw datasets. The unit-test command in the package README does not read these raw files, but extended data-fit and plotting workflows require the named inputs.

| Analysis | Expected local files | Source/notes |
|---|---|---|
| Planck TT reference | `data/COM_PowerSpect_CMB-TT-binned_R3.01.txt` | Planck Legacy Archive product R3.01; source URL is cited in the proposal. |
| Pantheon+ background fit | `data/Pantheon+SH0ES.dat`; `data/Pantheon+SH0ES_STAT+SYS.cov` | Obtain from the official Pantheon+ release and retain its attribution/terms. |
| DESI DR2 BAO fit | `data/desi_gaussian_bao_ALL_GCcomb_mean.txt`; `data/desi_gaussian_bao_ALL_GCcomb_cov.txt` | Obtain from the official DESI DR2 release and retain its attribution/terms. |
| SPARC rotation curves | `data/Rotmod_LTG.zip` or the selected Rotmod table | Official SPARC archive: https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip |
| Solar radial validation | `data/solar_density_profile_aag21.csv` | B23/SF-III AAG21 (Herrera & Serenelli 2023, v1.2), DOI: https://doi.org/10.5281/zenodo.10822316. Check the record's license before redistribution. |

The 119-test core command checks the packaged code paths without these files. Data-backed fit/solar tests and figure-generation workflows require the corresponding inputs. Confirm each source's current license and redistribution policy before adding raw files to a public repository or Zenodo record.

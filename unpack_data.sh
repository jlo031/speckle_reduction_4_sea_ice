
# run this in the folder with zip files

# unzip all folders
for f in `ls *.zip`; do
    unzip $f
done


# rename files
mv denoised_S1A_EW_GRDM_1SDH_20220502T074527_20220502T074631_043029_05233F_7BC7.tiff S1A_EW_GRDM_1SDH_20220502T074527_20220502T074631_043029_05233F_7BC7_ines_denoised.tiff
mv denoised_S1A_EW_GRDM_1SDH_20220502T074631_20220502T074731_043029_05233F_CC06.tiff S1A_EW_GRDM_1SDH_20220502T074631_20220502T074731_043029_05233F_CC06_ines_denoised.tiff
mv denoised_S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89.tiff S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_ines_denoised.tiff
mv denoised_S1A_EW_GRDM_1SDH_20220503T082725_20220503T082825_043044_0523D1_DCC4.tiff S1A_EW_GRDM_1SDH_20220503T082725_20220503T082825_043044_0523D1_DCC4_ines_denoised.tiff
mv S1A_EW_GRDM_1SDH_20220502T074527_20220502T074631_043029_05233F_7BC7.tiff S1A_EW_GRDM_1SDH_20220502T074527_20220502T074631_043029_05233F_7BC7_ines.tiff
mv S1A_EW_GRDM_1SDH_20220502T074631_20220502T074731_043029_05233F_CC06.tiff S1A_EW_GRDM_1SDH_20220502T074631_20220502T074731_043029_05233F_CC06_ines.tiff
mv S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89.tiff S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_ines.tiff
mv S1A_EW_GRDM_1SDH_20220503T082725_20220503T082825_043044_0523D1_DCC4.tiff S1A_EW_GRDM_1SDH_20220503T082725_20220503T082825_043044_0523D1_DCC4_ines.tiff


# move up one level
mv *tiff ../.


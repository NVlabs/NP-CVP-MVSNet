DTU_TEST_ROOT="../dataset/dtu-ours-hires-test-1200"
DEPTH_FOLDER="../out/np-cvp-mvsnet_model_000001.ckpt/"
OUT_FOLDER="fusibile_disp05"
FUSIBILE_EXE_PATH="./fusibile"

python2 -m pdb depthfusion.py \
--dtu_test_root=$DTU_TEST_ROOT \
--depth_folder=$DEPTH_FOLDER \
--out_folder=$OUT_FOLDER \
--fusibile_exe_path=$FUSIBILE_EXE_PATH \
--prob_threshold=0.8 \
--disp_threshold=0.5 \
--num_consistent=3 \
--image_height=1200

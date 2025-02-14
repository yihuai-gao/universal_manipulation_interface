target_dir="/scratch/m000073/yihuai/robotics/repositories/policies/diffusion_policy"

cp diffusion_policy/dataset/base_lazy_dataset.py $target_dir/unified_video_action/dataset/base_lazy_dataset.py
cp diffusion_policy/dataset/umi_lazy_dataset.py $target_dir/unified_video_action/dataset/umi_lazy_dataset.py
cp diffusion_policy/dataset/umi_multi_dataset.py $target_dir/unified_video_action/dataset/umi_multi_dataset.py

cp diffusion_policy/config/task/dataset/umi_lazy_dataset.yaml $target_dir/unified_video_action/config/task/umi_lazy.yaml
cp diffusion_policy/config/task/dataset/umi_multi_dataset.yaml $target_dir/unified_video_action/config/task/umi_multi.yaml

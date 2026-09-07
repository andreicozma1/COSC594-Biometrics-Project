# Contender Datasets

This page records datasets considered for the project and serves as a reference for selecting training and evaluation datasets.

Resolution notes distinguish dataset images, source recordings, and model preprocessing. Unverified dimensions are marked as uncertain.

## 1. Market-1501

Market-1501 contains RGB JPEG person crops from six outdoor campus surveillance cameras. Most crops were detected with DPM; query images were cropped manually.

- It covers 1,501 identities, with 32,668 detected crops and 3,368 query images counted separately. An optional set adds 500,000 distractor images.
- Training: 751 identities and 12,936 crops.
- Evaluation: 750 identities, 3,368 queries, and 19,732 gallery crops. Query and gallery items are individual crops, used for person retrieval across camera views.
- Crops are normalized to **64 × 128 pixels (width × height)**, according to Figure 1 of the paper.
- Annotations identify the person, camera, capture sequence/frame, and junk or distractor status. An attribute extension adds 27 identity-level attributes.
- Camera, time, and scene separation between the identity groups is not established by these notes.

### Project considerations

- Could serve as a supervised RGB ReID training set or an image-retrieval baseline.
- For a crop-based RGB model, load the released crops and preserve identity, camera, and query/gallery metadata.
- Use the selected evaluation code’s same-camera and junk exclusions when reporting benchmark results.

### Links

- Paper: [Scalable Person Re-identification: A Benchmark](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/ICCV15-ReIDDataset.pdf).
- Project page: [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html).
- GitHub repo: [Market-1501 Attribute](https://github.com/vana77/Market-1501_Attribute).

## 2. MSMT17

MSMT17 contains RGB person crops detected with Faster R-CNN across 12 outdoor and three indoor campus cameras.

- The paper describes 4,101 identities and 126,441 crops.
- Training: 1,041 identities and 32,621 images.
- Evaluation: 3,060 identities and 93,820 images, comprising 11,659 queries and 82,161 gallery images. The query/gallery task evaluates person-crop retrieval.
- The 180 camera-hours of source footage span four days within one month, with morning, noon, and afternoon recording periods.
- Crop resolution is **uncertain**; would need to check the dataset download.
- The release includes manually assigned identities, camera labels, and split lists.
- Camera and recording-period separation between training and evaluation is not established by these notes.

### Project considerations

- Could support supervised RGB ReID training and evaluation across varied campus capture conditions.
- For a crop-based RGB model, load crops using the release-specific split lists and retain camera metadata.
- To test indoor/outdoor or time-of-day robustness separately, define corresponding evaluation subsets.

### Links

- Paper: [Person Transfer GAN to Bridge Domain Gap for Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Person_Transfer_GAN_CVPR_2018_paper.html).
- Dataset page: [PKU VMC](https://www.pkuvmc.com/dataset/dataset.html).
- Access: [Release agreement](https://www.pkuvmc.com/agreement/RELEASE_AGREEMENT-MSMT17.pdf).
- GitHub repo: [PTGAN](https://github.com/pkuvmc/PTGAN).

## 3. CUHK03

CUHK03 provides manually labeled and automatically detected RGB person crops from campus surveillance. Each person appears in two camera views.

- The early dataset page reports 1,360 identities and 13,164 images; the later CUHK03-NP protocol uses 1,467 identities.
- CUHK03-NP assigns 767 identities to training and 700 to evaluation. Evaluation retrieves person crops across camera views.
- Labeled crops: 7,368 training images, 1,400 queries, and 5,328 gallery images.
- Detected crops: 7,365 training images, 1,400 queries, and 5,332 gallery images.
- The dataset and protocol are distributed as MATLAB files; later implementations export individual image files.
- The paper preprocesses images to **64 × 32 × 3** (including color channels); original crop dimensions would need to be verified.
- Camera and recording-time separation is unverified.

### Project considerations

- Could provide a supervised RGB ReID baseline and a comparison between manually labeled and automatically detected crops.
- Export or load the MATLAB data, choose one crop variant, and pair it with the matching CUHK03-NP protocol files.
- When comparing crop variants, hold the identity protocol, model, and evaluation procedure fixed.

### Links

- Paper: [DeepReID: Deep Filter Pairing Neural Network for Person Re-identification](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html).
- Dataset page: [CUHK person-identification datasets](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html).
- GitHub repo: [Person re-ranking](https://github.com/zhunzhong07/person-re-ranking), including the [CUHK03-NP protocol](https://github.com/zhunzhong07/person-re-ranking/blob/master/CUHK03-NP/README.md).

## 4. CUHK-SYSU

CUHK-SYSU supports person search across uncropped RGB scene galleries drawn from street photography and movie/TV frames.

- Its 18,184 images contain 96,143 person boxes and 8,432 identified people.
- Street photography contributes 12,490 images with 6,057 identities; movie/TV sources contribute 5,694 images with 2,375 identities.
- Scene-image resolution is **uncertain**; would need to check the dataset download.
- Training: 11,206 images, 55,272 boxes, and 5,532 identities.
- Evaluation: 6,978 images, 40,871 boxes, 2,900 identities, and 2,900 queries.
- The common gallery protocol uses 100 scene images per query, with additional gallery-size settings available. Each scene can contain multiple person instances; the task searches these scenes for a query person.
- Annotations include person boxes and identity labels, including people without known identities. The described annotation policy ignores background pedestrians below 50 pixels tall.
- Street-versus-movie separation and independence of underlying source material across splits are not established.

### Project considerations

- Could support person-search experiments or provide labeled person crops for a supervised RGB model.
- For a crop-based model, extract annotated person instances and retain scene, box, and identity mappings; retain full scenes for the native person-search protocol.
- Evaluation on extracted ground-truth gallery crops would measure a different task from native person search. Exclude unknown identity labels from supervised person classes unless a separate labeling procedure establishes their identities.

### Links

- Paper: [Joint Detection and Identification Feature Learning for Person Search](https://arxiv.org/abs/1604.01850).
- Dataset page: [CUHK-SYSU](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html).
- GitHub repo: [Person search](https://github.com/ShuangLI59/person_search).

## 5. PRW

PRW provides full RGB surveillance frames from six campus cameras, plus query person crops.

- It contains 11,816 frames, 43,110 person boxes, and 932 known identities.
- The project reports that 34,304 boxes have known identities.
- Training: 5,704 frames and 482 identities.
- Evaluation: 6,112 frames, 450 identities, and 2,057 query crops. Native evaluation searches full surveillance frames, each of which can contain multiple people.
- Approximately ten hours of source video were recorded at 25 fps and sampled for annotation at one frame per second.
- Five cameras record at 1920 × 1080 and one at 720 × 576.
- Query crops have **variable dimensions** and are not normalized, according to the project page.
- MATLAB annotation files contain boxes and identity labels, with -2 marking unknown identities.
- The listed release is 2.67 GB.
- Camera and source-video-period separation between training and evaluation is not established.

### Project considerations

- Could support person-search evaluation or supervised learning from annotated surveillance person crops.
- For a crop-based model, read the MATLAB boxes, extract person crops, and preserve frame, camera, and identity mappings.
- Keep -2 out of supervised identity classes; it denotes unknown people, not one shared person. Scoring pre-extracted gallery crops would change the native evaluation task.

### Links

- Paper: [Person Re-identification in the Wild](https://arxiv.org/abs/1604.02531).
- Project page: [PRW](https://zheng-lab-anu.github.io/Project/project_prw.html).
- GitHub repo: [PRW baseline](https://github.com/liangzheng06/PRW-baseline).

## 6. MARS

MARS contains RGB JPEG person-crop sequences produced by DPM detection and GMMCP tracking across six campus cameras. The standard package provides tracklet images rather than full-scene videos.

- It covers 1,261 identities, 20,478 tracklets, and 1,191,003 crops; the project also reports 3,248 distractor tracklets.
- Training: 625 identities, 8,298 tracklets, and 509,914 crops.
- Evaluation: 636 identities, 12,180 tracklets, and 681,089 crops across query and gallery.
- Crop resolution is **uncertain**; would need to check the dataset download.
- Annotations include identity, camera, tracklet, frame ordering, and junk/distractor labels. Person identity, tracklet identity, and distractor status have distinct meanings.
- The release is 6.26 GB.
- The retrieval unit is an ordered tracklet of person crops; its frames are related observations.
- Camera separation between the training and evaluation identity groups is not established.

### Project considerations

- Could support supervised video ReID and controlled comparisons of single-frame and temporal models.
- Choose a frame-sampling and sequence-aggregation policy, preserve ordering, and apply the intended junk/distractor filtering protocol.
- To assess temporal aggregation, compare against a suitable frame-based baseline and keep frames from the same tracklet together when constructing splits.

### Links

- Paper: [MARS: A Video Benchmark for Large-Scale Person Re-identification](https://mlanthology.org/eccv/2016/zheng2016eccv-mars/).
- Project page: [MARS](https://zheng-lab-anu.github.io/Project/project_mars.html).
- GitHub repo: [MARS evaluation](https://github.com/liangzheng06/MARS-evaluation).

## 7. LS-VID

LS-VID contains RGB person-crop sequences generated from surveillance videos using detection and tracking.

- Its 3,772 identities appear in 14,943 sequences containing 2,982,685 cropped frames, averaging approximately 200 frames per sequence.
- The 180 camera-hours of source footage come from 15 cameras, comprising 12 outdoor and three indoor cameras, across four recording days and 12 time periods.
- Training (introducing paper’s protocol): 842 identities and 550,419 cropped frames.
- Validation: 200 identities and 155,191 cropped frames.
- Evaluation: 2,730 identities and 2,277,075 cropped frames.
- Images are resized to **256 × 128** for model input; original crop dimensions would need to be verified.
- The video-retrieval unit is a sequence of ordered person crops. Person identities, sequences, and constituent frames are separate inventory units.
- Camera and recording-day separation, along with exact query/gallery sequence membership, remain unverified.

### Project considerations

- Could support supervised video ReID with a separate validation set for model selection.
- Load sequence metadata, define frame sampling and aggregation, and preserve the introducing paper’s train/validation/evaluation protocol.
- Derive frame labels from the released sequence metadata. Separate tests of recording-day or indoor/outdoor robustness require corresponding evaluation subsets.

### Links

- Paper: [Global-Local Temporal Representations for Video Person Re-Identification](https://arxiv.org/abs/1908.10049).
- Dataset page: [PKU VMC](https://www.pkuvmc.com/dataset/dataset.html).
- Access: [Release agreement](https://www.pkuvmc.com/agreement/RELEASE%20AGREEMENT-LS-VID.pdf).
- GitHub repo: [GLTR](https://github.com/rika1024/GLTR).

## 8. MEVID

MEVID provides RGB person-chip sequences, annotations, check-in photographs, and separately available full source-video clips from MEVA.

- It covers 158 people, 598 outfits, and 8,092 tracklets across 33 views at 17 locations.
- Indoor and outdoor recordings span nine dates within a 73-day window.
- Training: 104 identities, 485 outfits, and 6,338 tracklets.
- Evaluation: 54 identities, 113 outfits, and 1,754 tracklets.
- The 976 five-minute source clips total 81 hours 20 minutes, calculated from the published clip count and length.
- Annotations include identity, outfit, camera, image ordering, and tracklet boundaries. Person identity persists across outfits; outfit and tracklet identifiers describe different properties.
- Training chips occupy 30.5 GB, test chips 13 GB, source videos 127 GB, and check-in photographs 600 MB.
- Person-chip resolution is **uncertain**; would need to check the dataset download.
- A ReID sample is an ordered person-chip tracklet.
- The inventory does not specify which query/gallery pairs isolate outfit or viewpoint changes, or establish location, date, and outfit independence across person splits.

### Project considerations

- Could support video ReID and evaluation involving outfit, viewpoint, and recording-date variation.
- Start from the chips for a crop-based sequence model, preserve outfit and camera metadata, and define frame sampling; source-video processing is a separate choice.
- Select query/gallery pairs explicitly when testing outfit or viewpoint changes, rather than attributing aggregate performance to either factor.

### Links

- Paper: [MEVID: Multi-view Extended Videos with Identities for Video Person Re-Identification](https://arxiv.org/abs/2211.04656).
- Project page: [MEVA](https://mevadata.org/index.html).
- GitHub repo: [MEVID](https://github.com/Kitware/MEVID).

## 9. P-DESTRE

P-DESTRE provides full RGB drone videos, person ROI crops, and frame-level annotations from crowded outdoor university scenes in Portugal and India.

- A DJI Phantom 4 recorded from approximately 5.5–6.7 m altitude, with camera pitch ranging from oblique to overhead views.
- Videos use H.264 MP4 at 3840 × 2160, 30 fps, and 24-bit color.
- ROI-crop resolution is **uncertain**; would need to check the dataset download. The dimensions above describe the source videos.
- The paper’s comparison table reports over 14.8 million boxes.
- Identity-related figures differ: 253 in that table, 261 in the paper’s description, and 269 volunteers on the download page.
- Annotations include frame-level boxes, track/identity labels, cross-day identity information, and 16 soft-biometric attributes. Later head-pose values are estimates.
- The ReID protocol uses five repeated splits with 50% training, 10% query, and 40% gallery data. These proportions do not establish whether partitioning occurs by person, track, frame, or recording day.
- Videos occupy 38.9 GB and ROI crops 30.5 GB.
- No maintained dataset repository found; the project provides data and evaluation resources.

### Project considerations

- Could support aerial person ReID and experiments using temporal or soft-biometric information.
- Use the ROI crops or extract them from video, preserve frame/track associations, and reproduce the chosen repeated-split protocol.
- Choose whether our model consumes individual ROI crops or sequences assembled from track annotations. Verify the partition unit and identity mapping, and select pairs explicitly for cross-day or viewpoint comparisons.

### Links

- Paper: [The P-DESTRE Dataset](https://arxiv.org/abs/2004.02782).
- Project page: [P-DESTRE](https://p-destre.di.ubi.pt/index.html).
- Downloads: [Videos, annotations, and crops](https://p-destre.di.ubi.pt/download.html).

## 10. UAV-Human

UAV-Human includes drone-recorded multimodal action videos and separate person-crop ReID, pose, and attribute datasets.

- Moving UAV viewpoints cover indoor/outdoor and day/night conditions at 45 locations over three months.
- The action dataset contains 67,428 video sequences across three synchronized sensor streams, with 119 subjects and 155 action classes. Modalities include RGB, depth, infrared, fisheye, and night vision.
- The ReID dataset contains 41,290 person crops and 1,144 benchmark identities constructed from person and recording-setup identifiers. This benchmark identity count is not a count of unique human subjects.
- ReID training: 619 identities and 11,805 crops.
- ReID evaluation: 525 identities, 1,050 queries, and 28,435 gallery crops. The task evaluates person-crop retrieval under UAV capture conditions.
- Pose data contains 22,476 frames with 17 joints; attribute data contains 22,263 frames and seven attribute groups.
- RGB resolution is 1920 × 1080, depth/infrared is 640 × 576, and fisheye/night vision is 640 × 480.
- ReID-crop resolution is **uncertain**; would need to check the dataset download. The dimensions above describe the source recordings.
- Separation of underlying people, recording setups, and locations across benchmark identity groups is unverified.

### Project considerations

- Could support UAV person-crop ReID; action, pose, and attribute learning would be separate project choices.
- Load the ReID crops and split metadata; use a separate mapping if introducing pose, attributes, or other modalities.
- Keep the ReID inventory separate from the action, pose, and attribute components; action-dataset modality and scene diversity does not establish equivalent coverage in the ReID protocol.

### Links

- Paper: [UAV-Human paper](https://arxiv.org/abs/2104.00946).
- Project page: [UAV-Human project page](https://sutdcv.github.io/uav-human-web/).
- GitHub repo: [UAV-Human repository](https://github.com/SUTDCV/UAV-Human).

## 11. G2APS

G2APS provides full RGB ground-camera and aerial frames for ground-to-aerial person search. It is distinct from G2A-VReID.

- The dataset contains 31,770 images evenly divided between ground and aerial views, 260,559 person boxes, and 2,644 known identities across nine scenes.
- Frames were sampled at 2 fps from 36 source videos, with UAV altitude approximately 20–60 m.
- Ground and aerial frame resolutions are **uncertain**; would need to check the dataset download.
- Of the annotated boxes, 199,696 have known identities and 60,863 have unknown identities.
- Training: 21,962 images and 2,078 identities, according to Table 3 and the implementation setup.
- Evaluation: 9,808 images, 566 identities, and 566 ground-image queries. The described gallery contains 50 aerial frames per query. Each aerial gallery frame can contain multiple people.
- Separation by scene or source video across the training and evaluation groups is not established.

### Project considerations

- Could support ground-to-aerial person search or a crop-based experiment on platform changes.
- For a crop-based model, extract annotated instances and preserve platform, scene, box, and identity mappings; retain full scenes for native evaluation.
- A reverse-direction or crop-only experiment needs its own declared protocol. Unknown boxes do not establish additional supervised person classes.

### Links

- Paper: [Ground-to-Aerial Person Search: Benchmark Dataset and Approach](https://arxiv.org/abs/2308.12712).
- GitHub repo: [HKD for Person Search](https://github.com/yqc123456/HKD_for_person_search), including the dataset download.

## 12. SYSU-MM01

SYSU-MM01 contains RGB and near-infrared person crops from four RGB and two infrared cameras.

- Its standard crop release covers 491 identities, with 30,071 RGB crops and 15,792 infrared crops, totaling 45,863 images.
- Separate Kinect RGB source videos are also described.
- The larger figure of 287,628 RGB images includes additional RGB video frames and should be distinguished from the standard crop count.
- RGB and infrared crop resolutions are **uncertain**; would need to check the dataset download.
- The identity split assigns 296 to training, 99 to validation, and 96 to testing. Common implementations combine training and validation into 395 training identities.
- Evaluation uses 3,803 infrared queries and randomly sampled RGB galleries, with single-shot/ten-shot and indoor-search/all-search protocols.
- Access requires a signed agreement through the dataset repository.
- Person identities link observations across cameras and modalities. Each retrieval item is one modality-specific crop; synchronized or pixel-aligned RGB/infrared pairs are not established by a shared identity.

### Project considerations

- Could support supervised RGB–infrared representation learning and cross-modality retrieval evaluation.
- Load both crop modalities, preserve camera labels, and reproduce the gallery sampling and exclusion rules of the selected evaluator.
- Report the selected indoor/all-search and single-/ten-shot settings. If training and validation identities are combined, declare a separate model-selection procedure.

### Links

- Papers: [RGB-Infrared Cross-Modality Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_RGB-Infrared_Cross-Modality_Person_ICCV_2017_paper.html); [IJCV extension](https://www.eecs.qmul.ac.uk/~sgg/papers/AncongWuEtAl_IJCV2020.pdf).
- GitHub repositories: [SYSU-MM01](https://github.com/wuancong/SYSU-MM01); [Python evaluation](https://github.com/InnovArul/SYSU_MM01_pythoneval).

## 13. RegDB

RegDB contains paired visible-light and thermal person crops from aligned cameras.

- Each of its 412 identities has ten visible and ten thermal images, yielding 4,120 images per modality and 8,240 in total.
- Training and testing each use 206 identities, with 2,060 visible and 2,060 thermal images in each half.
- Evaluation covers visible-to-thermal and thermal-to-visible retrieval, commonly across ten randomized trials.
- Visible and thermal crop resolutions are **uncertain**; would need to check the dataset download.
- Annotations provide identity and modality labels alongside trial-specific split files. Person labels link the two modalities, and trial files assign training and test identities.
- A retrieval item is one visible or thermal person crop. Recording-session separation and statistical independence of repeated trials are not established by the split sizes.

### Project considerations

- Could provide a compact supervised visible–thermal training and retrieval benchmark.
- Load both modalities, retain their labels, and use the same trial definitions when comparing methods or retrieval directions.

### Links

- Paper: [Person Recognition System Based on a Combination of Body Images from Visible Light and Thermal Cameras](https://www.mdpi.com/1424-8220/17/3/605).
- Dataset page: [Dongguk database page](http://dm.dongguk.edu/link.html), a legacy access route.
- GitHub repo: [Visible-Thermal Person Re-Identification](https://github.com/mangye16/Visible-Thermal-Person-Re-Identification), a follow-up baseline and access reference.

## 14. WILDTRACK

WILDTRACK provides full RGB frames from seven synchronized, calibrated, overlapping surveillance cameras overlooking an outdoor pedestrian area.

- The annotated portion contains 400 synchronized timestamps and 2,800 camera images, covering approximately 200 seconds at 2 annotated timestamps per second.
- The 2018 paper reports more than 300 individuals and over 40,000 boxes.
- Released, distortion-corrected frames measure **1920 × 1080 pixels (width × height)**, according to the project page.
- Source recording runs at 60 fps, additional extracted imagery at 10 fps, and the annotated benchmark at 2 fps.
- The common split uses the first 360 synchronized timestamps for training and the final 40 for testing. This is a temporal split; separation by person identity, camera, or scene is not established.
- Annotations include cross-view person identities, image-space boxes, ground-plane positions, and camera calibration.
- One synchronized observation groups seven camera images. The native annotations support multi-view localization and association.

### Project considerations

- Could support experiments using synchronized camera views or provide annotated crops for a custom ReID study.
- Extract person crops while retaining timestamp and camera alignment; calibration and ground-plane coordinates matter if the method uses geometry.
- A query/gallery ReID study would need an explicitly constructed protocol, including a decision about whether evaluation should use unseen identities.

### Links

- Papers: [WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html); [earlier dataset paper](https://arxiv.org/abs/1707.09299).
- Dataset page: [EPFL WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/).
- GitHub repositories: [WILDTRACK toolkit](https://github.com/Chavdarova/WILDTRACK-toolkit); [multi-camera annotation tools](https://github.com/cvlab-epfl/multicam-gt).

## 15. MOT17

MOT17 provides full RGB video-frame sequences from static and moving ground cameras for single-camera pedestrian tracking.

- It has 14 distinct sequences, divided into seven training and seven test sequences. DPM, Faster R-CNN, and SDP variants reuse the imagery. The train/test partition is by sequence.
- Training: 5,316 frames; Table 9 of the benchmark paper reports 546 trajectories and 112,297 pedestrian boxes.
- Test: 5,919 frames; the same table reports 785 trajectories and 188,076 boxes.
- The combined total is 11,235 unique frames.
- Trajectories are sequence-local rather than globally unique people. Separation of the underlying people across train/test sequences is not established.
- Frame rates are 14, 25, or 30 fps, with sequences lasting approximately 15–85 seconds and about 7 minutes 43 seconds combined.
- Full frames measure **640 × 480 pixels** for MOT17-05 and MOT17-06, and **1920 × 1080** for the other sequences (width × height), according to Table 9 of the benchmark paper.
- The MOT17 package is 5.5 GB; the separate MOT17Det package is 1.9 GB.

### Project considerations

- Could support tracking evaluation or supply track-associated crops for an auxiliary representation-learning experiment.
- For a crop-based component, extract boxes from the appropriate annotations and retain sequence/frame/trajectory mappings; account for imagery reused across detector variants.
- Namespace trajectory IDs by sequence and keep detector variants of the same imagery together. A crop-retrieval study needs a separate query/gallery protocol from native tracking evaluation.

### Links

- Paper: [MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking](https://arxiv.org/abs/2010.07548).
- Dataset pages: [MOT17](https://motchallenge.net/data/MOT17/); [MOT17Det](https://motchallenge.net/data/MOT17Det/).
- GitHub repo: [TrackEval](https://github.com/JonathonLuiten/TrackEval).

## 16. COCAS

COCAS pairs RGB person crops with clothing-template images for clothes-changing ReID.

- It contains 62,382 body images of 5,266 identities from 30 cameras, covering indoor/outdoor scenes and up to three outfits per identity.
- Training: 2,800 identities and 34,019 images, calculated by subtracting the published query and gallery totals.
- Evaluation: 2,466 identities, 15,985 queries, and 12,378 gallery images.
- Body-crop and clothing-template resolutions are **uncertain**; would need to check the selected dataset download.
- Annotations include identity, clothing/template associations, and query/gallery protocols. Person identity persists across the associated clothing changes.
- COCAS-plus Real2 contains 101 identities and 21,319 images, while its synthetic subset contains 1,038 identities and 92,526 images. These extensions are separate datasets, not replacements for the base inventory.
- The intended evaluation uses a specified clothing-template/query arrangement. Camera separation is unverified.

### Project considerations

- Could support clothes-changing ReID, particularly methods that use clothing-template information.
- Load body crops, clothing associations, and query/gallery lists; decide explicitly how clothing templates enter the model.
- If using person crops without clothing-template inputs, document how the experiment differs from the selected template/query protocol. Establish extension-specific splits if using COCAS-plus.

### Links

- Paper: [COCAS: A Large-Scale Clothes Changing Person Dataset for Re-Identification](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_COCAS_A_Large-Scale_Clothes_Changing_Person_Dataset_for_Re-Identification_CVPR_2020_paper.html).
- GitHub repo: [COCAS-plus](https://github.com/Chenhaobin/COCAS-plus), including extension downloads.

## 17. LTCC

LTCC contains RGB PNG person crops from long-term surveillance across 12 cameras over two months.

- Its 152 identities appear in 17,119 images.
- Among them, 91 identities have clothing changes across 416 outfits and 14,783 images; the other 61 clothing-consistent identities account for 2,336 images.
- Training: 77 identities and 9,576 images.
- Evaluation: 75 identities, 493 queries, and 7,050 gallery images.
- Crop resolution is **uncertain**; would need to check the dataset download.
- Annotations provide identity, camera, and clothing labels, plus lists of identities with changing or consistent clothing. Person identities persist across outfits.
- The exact query/gallery clothing and camera exclusions remain protocol-dependent; the collection window does not establish chronological or camera separation in the test split.

### Project considerations

- Could support supervised training and evaluation focused on clothing changes in long-term surveillance.
- Load crops with their clothing and camera labels, preserve the identity-group lists, and implement the chosen evaluation filters.

### Links

- Paper: [Long-Term Cloth-Changing Person Re-identification](https://arxiv.org/abs/2005.12633).
- Project page: [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html).
- Access: [Release agreement](https://naiq.github.io/images/arxiv2020_LTCC/Restrictions%20for%20The%20Use%20of%20LTCC%20Dataset.pdf).
- GitHub repo: [FIRe-CCReID](https://github.com/QizaoWang/FIRe-CCReID), a follow-up method linked by the project.

## 18. PRCC

PRCC provides RGB person crops and derived contour sketches from three indoor cameras, with 221 identities and 33,698 images.

- Cameras A and B show the same clothing in different rooms; camera C shows changed clothing on another day.
- Contour sketches are resized to **224 × 224** for model input; original RGB crop and contour dimensions would need to be verified.
- The paper assigns 150 identities to training and 71 to testing, reserving a portion of the training imagery for validation.
- Evaluation compares same-clothes matching from B to A with clothes-changing matching from C to A. Implementations sample galleries across repeated trials. The changed-clothes direction also changes camera and recording day.
- Annotations include identity, camera, and clothing-condition information. Person identity links observations across those conditions.
- The camera roles in query/gallery construction do not establish unseen training cameras.

### Project considerations

- Could support comparisons between same-clothes and changed-clothes ReID, with RGB or contour-based models.
- Choose RGB, contours, or both, then reproduce the repeated gallery sampling and the selected retrieval direction.
- Interpret the same-clothes versus changed-clothes comparison as a combined change in clothing, camera, and day; it cannot isolate clothing alone.

### Links

- Paper: [Person Re-identification by Contour Sketch under Moderate Clothing Change](https://arxiv.org/abs/2002.02295).
- Project page: [PRCC](https://www.isee-ai.cn/~yangqize/clothing.html).
- GitHub repo: [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID), including a [PRCC dataset loader](https://github.com/guxinqian/Simple-CCReID/blob/main/data/datasets/prcc.py).

## 19. LUPerson

LUPerson contains automatically detected RGB person crops from YouTube street-view videos across diverse cameras, cities, and environments.

- It includes 4,180,243 crops from 46,260 scenes.
- The collection pipeline retained 50,534 source videos after initial filtering.
- The reported population of 219,848 people is an estimate obtained by summing the maximum visible-person count per video, not a verified unique-identity count.
- Frames are sampled every 100 frames; YOLOv5 person detections are then filtered by visibility, confidence, size, and aspect ratio.
- Crop dimensions are **variable**, according to Table 1 of the paper.
- Crops must be wider than 48 pixels, with height/width between 1.5 and 5.
- LUPerson is intended for unlabeled pretraining and has no standard identity-labeled ReID query/gallery split.
- LUPerson-NL is a separate extension with different label assumptions; the base population estimate does not supply supervised identity labels.

### Project considerations

- Could support unsupervised pretraining before supervised training on an identity-labeled dataset.
- Use the released or reconstructed crops with a declared pretraining pipeline; reconstruction completeness must be checked if following construction instructions.
- Assess the pretrained representation on a separate identity-labeled benchmark. Any custom validation split should account for shared source videos and possible person overlap.

### Links

- Paper: [Unsupervised Pre-training for Person Re-Identification](https://arxiv.org/abs/2012.03753).
- GitHub repositories: [LUPerson](https://github.com/DengpanFu/LUPerson); [construction instructions](https://github.com/DengpanFu/LUPerson/tree/main/LUP); [LUPerson-NL](https://github.com/DengpanFu/LUPerson-NL), a separate noisy-label extension.

## 20. LaST

LaST contains RGB person crops extracted from more than 2,000 movies spanning multiple countries, with long spatial/temporal gaps and clothing changes.

- The repository/Table II inventory reports 10,862 identities and 228,156 images.
- Training: 5,000 identities and 71,248 images.
- Validation: 56 identities, 100 queries, and 21,279 gallery images.
- Evaluation: 5,806 gallery identities in 125,353 gallery images, plus 10,176 queries covering 5,805 identities. The task is person-crop retrieval within the dataset’s movie-derived variation.
- Images are resized to **256 × 128** for model input; original crop dimensions would need to be verified.
- Annotations include identity labels, split lists, and training clothing labels.
- Separation of movie sources, actors, or recording scenes across the identity groups is not established by the notes.

### Project considerations

- Could support supervised RGB ReID under clothing changes and long spatial or temporal gaps in movie imagery.
- Load the crop lists and clothing metadata, preserving the repository/Table II inventory and its separate validation split.
- To isolate clothing or temporal effects, define suitable subsets or controls rather than attributing aggregate retrieval performance to either factor.

### Links

- Paper: [Large-Scale Spatio-Temporal Person Re-identification: Algorithms and Benchmark](https://arxiv.org/abs/2105.15076).
- Project page: [LaST project page](https://sites.google.com/view/personreid).
- GitHub repo: [LaST repository](https://github.com/shuxjweb/last), including download links.

## 21. Gait3D

Gait3D represents person gait sequences as silhouettes, pose/keypoint data, and estimated 3D human representations. Raw RGB footage is withheld from the standard release.

- It contains 4,000 identities, 25,309 sequences, and over three million person-frame observations.
- Source video was recorded at 1920 × 1080 and 25 fps across 39 supermarket cameras over seven days, totaling approximately 1,090 camera-hours.
- Silhouettes retain **variable original dimensions and aspect ratios**. The paper uses **88 × 128** and **44 × 64 pixels (width × height)** for model input; pose and SMPL data are not raster images.
- Most sequences exceed 50 frames; the cap of 500 frames corresponds to 20 seconds at the source frame rate.
- The split uses 3,000 training identities and 1,000 test identities.
- Evaluation uses 1,000 query sequences and 5,369 gallery sequences. Identity labels connect sequences belonging to a person; frames within a sequence are related observations.
- The release size is 150 GB reported by the [CVF dataset catalog](https://cove.thecvf.com/datasets/777); not independently measured.
- Camera and recording-day separation between training and test identity groups is not established.

### Project considerations

- Could support supervised gait recognition or a study combining the released gait representations.
- Choose a released representation and compatible sequence model, then define temporal sampling. A conventional RGB-crop model would require inputs absent from the standard release.
- When comparing representations, hold the sequence membership and evaluation protocol fixed so the input representation is the intended difference.

### Links

- Paper: [Gait Recognition in the Wild with Dense 3D Representations and a Benchmark](https://arxiv.org/abs/2204.02569).
- Project page: [Gait3D](https://gait3d.github.io/).
- GitHub repo: [Gait3D Benchmark](https://github.com/Gait3D/Gait3D-Benchmark).
- Access: [Release agreement](https://gait3d.github.io/resources/AgreementForGait3D.pdf).

## 22. GREW

GREW contains outdoor surveillance gait sequences distributed as silhouettes, gait energy images, optical flow, and 2D/3D poses. Raw RGB footage is withheld.

- The dataset covers 26,345 identities, 128,671 sequences, and 14,185,478 person frames.
- Training: 20,000 identities, 102,887 sequences, and 10,166,842 frames.
- Validation: 345 identities, 1,784 sequences, and 238,532 frames.
- Evaluation: 6,000 identities, 24,000 sequences, and 3,780,104 frames, with two query and two gallery sequences per identity. The retrieval unit is a person sequence in the selected representation.
- An additional distractor set contains 233,857 sequences and 9,676,016 frames. This inventory does not establish additional labeled training identities.
- Collection spans 882 cameras at over 600 locations, with 7,533 source clips drawn from approximately 3,500 hours of 1080p footage.
- Released silhouettes and gait energy images measure **64 × 64 pixels**, according to the repository. Pose records retain the original image dimensions in their headers.
- Optical-flow image resolution is **uncertain**; would need to check the dataset download.
- Separation of source clips, locations, and cameras across the identity groups remains unverified.

### Project considerations

- Could support large-scale supervised gait training and sequence-retrieval evaluation with optional distractors.
- Select silhouettes, gait energy images, optical flow, or poses and use a compatible model; preserve sequence membership when sampling observations.
- Report whether the separate distractor set is included, since it changes the retrieval conditions.

### Links

- Paper: [Gait Recognition in the Wild: A Large-Scale Benchmark and NAS-Based Baseline](https://arxiv.org/abs/2205.02692).
- Project page: [GREW](https://www.grew-benchmark.org/).
- GitHub repo: [GREW Benchmark](https://github.com/GREW-Benchmark/GREW-Benchmark).

## 23. AerialGait

AerialGait covers aerial/ground person gait sequences.

- The paper describes person RGB crops, silhouettes, 2D/3D poses, and human-parsing representations for 533 subjects, 82,454 sequences, and 10,259,669 person frames.
- A moving DJI Mavic 3 drone and ground camera recorded across five environments.
- RGB-crop resolution is **uncertain**; would need to confirm crop availability and dimensions in the dataset release.
- The accessible submission describes 100 training subjects and 433 test subjects, using normal-walking gallery sequences and multiple aerial/ground retrieval directions.
- No maintained dataset download page found.
- No maintained dataset repository found.
- The inventory distinguishes human subjects, sequences, and person frames. Final-release split membership and separation of environments or sessions are unverified.

### Project considerations

- Could be a candidate for aerial-to-ground gait experiments if access to the described representations is established.
- First confirm the available package and protocol, then choose a compatible representation and define sequence sampling.

### Links

- Paper: [AerialGait: Bridging Aerial and Ground Views for Gait Recognition](https://doi.org/10.1145/3664647.3681002).
- Additional paper copy: [OpenReview submission](https://openreview.net/pdf?id=t87LMw4CpY).

## 24. MP-ReID

MP-ReID contains JPG person crops across ground and UAV platforms, with RGB, near-infrared, and thermal modalities in indoor/outdoor scenes.

- It covers 1,930 identities and 136,156 person images from 14 cameras: six ground RGB, six ground near-infrared, one UAV RGB, and one UAV thermal camera.
- Over 1.2 million source frames came from more than 13 hours of footage collected over four months.
- Ground cameras record at 1920 × 1080 and 25 fps; UAV RGB records at 3840 × 2160 and thermal at 640 × 512, both at 30 fps.
- UAV altitudes are 5, 7, and 10 m.
- Crop resolutions by platform and modality are **uncertain**; would need to check the dataset download. The dimensions above describe the source cameras.
- Six experiment-specific protocols have separate split files rather than one universal training/test division.
- Annotations include identity, camera/platform, and modality information. Person identity is separate from platform and modality; cross-modality pairing or synchronization is not established.
- Detections were manually checked and faces obscured.

### Project considerations

- Could support supervised ReID experiments involving both platform and modality changes.
- Choose a protocol, load its crop and split files, and preserve platform/modality metadata; adapt the model only for modalities included in that experiment.
- For each protocol, verify identity membership and query/gallery assignments, and state whether platform and modality change together.

### Links

- Paper: [Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](https://arxiv.org/abs/2503.17096).
- Project page: [MP-ReID project page](https://mp-reid.github.io/).
- GitHub repo: [MP-ReID repository](https://github.com/MP-ReID/mp-reid).
- Access: [Dataset instructions and agreement route](https://github.com/MP-ReID/mp-reid/blob/main/dataset.md).

## 25. RGBNT201

RGBNT201 contains aligned RGB, near-infrared, and thermal person crops from four camera views on campus, covering day, night, and fog conditions.

- It has 201 identities and 4,787 aligned three-modality records, equivalent to 14,361 individual images. A multimodal sample is one aligned record, with person identity linking records and correspondence linking its three modalities.
- The paper assigns 141 identities to training, 30 to validation, and 30 to testing.
- Evaluation uses ten query records per test identity under that protocol. The paper’s task is multimodal person retrieval.
- Collection spans four months and over 9,000 seconds, or 2.5 hours, of source footage.
- RGB/near-infrared sources record at 700 × 580 and 15 fps; thermal sources at 640 × 480 and 20 fps.
- The paper reports **256 × 128 pixels** for crops in each modality; width/height order would need to be verified.
- Camera and collection-period separation across the identity groups is unverified.

### Project considerations

- Could support supervised multimodal ReID and controlled comparisons of RGB, near-infrared, and thermal inputs.
- Load aligned triplets, preserve their correspondence during splitting and preprocessing, and choose which modalities the model consumes.
- When removing or combining modalities, keep all images from one record in the same split and hold record membership and evaluation conditions fixed.

### Links

- Paper: [Robust Multi-Modality Person Re-identification](https://ojs.aaai.org/index.php/AAAI/article/view/16467).
- Dataset page: [Author publication page](https://ziwang1121.github.io/publication.html).
- Download: [RGBNT201 distribution folder](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH).
- GitHub repo: [IEEE](https://github.com/ziwang1121/IEEE), a follow-up multimodal method linked by the author.

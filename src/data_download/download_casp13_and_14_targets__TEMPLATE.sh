

# saving the pdbs in two places because then I can parse the casp13 and casp14 folders separately to get the lists of names.
# but also I want them all in one spot so that I can process them with a single script that submits jobs


# download casp 13
curl https://predictioncenter.org/download_area/CASP13/targets/casp13.targets.T.4public.tar.gz > casp13.targets.T.4public.tar.gz
mkdir -p {BASE_DIR}/casp_targets/
mkdir -p {BASE_DIR}/casp13_targets/
tar -xvf casp13.targets.T.4public.tar.gz -C {BASE_DIR}/casp_targets/
tar -xvf casp13.targets.T.4public.tar.gz -C {BASE_DIR}/casp13_targets/
rm casp13.targets.T.4public.tar.gz

curl https://predictioncenter.org/download_area/CASP13/targets/_4predictors/casp13.targets.T.4predictors.tar.gz > casp13.targets.T.4predictors.tar.gz
mkdir -p {BASE_DIR}/casp_targets/
mkdir -p {BASE_DIR}/casp13_targets/
tar -xvf casp13.targets.T.4predictors.tar.gz -C {BASE_DIR}/casp_targets/
tar -xvf casp13.targets.T.4predictors.tar.gz -C {BASE_DIR}/casp13_targets/
rm casp13.targets.T.4predictors.tar.gz


# download casp 14
curl https://predictioncenter.org/download_area/CASP14/targets/casp14.targets.T.public_11.29.2020.tar.gz > casp14.targets.T.public_11.29.2020.tar.gz
mkdir -p {BASE_DIR}/casp_targets/
mkdir -p {BASE_DIR}/casp14_targets/
tar -xvf casp14.targets.T.public_11.29.2020.tar.gz -C {BASE_DIR}/casp_targets/
tar -xvf casp14.targets.T.public_11.29.2020.tar.gz -C {BASE_DIR}/casp14_targets/
rm casp14.targets.T.public_11.29.2020.tar.gz

curl https://predictioncenter.org/download_area/CASP14/targets/_4invitees/casp14.targ.whole.4invitees.tgz > casp14.targ.whole.4invitees.tgz
mkdir -p {BASE_DIR}/casp_targets/
mkdir -p {BASE_DIR}/casp14_targets/
tar -xvf casp14.targ.whole.4invitees.tgz -C {BASE_DIR}/casp_targets/
tar -xvf casp14.targ.whole.4invitees.tgz -C {BASE_DIR}/casp14_targets/
rm casp14.targ.whole.4invitees.tgz








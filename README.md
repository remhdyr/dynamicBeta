# dynamicBeta
Dynamice _beta_ Variational Auto Encoder (VAE) for biodiversity assessment of insect signals. 
A fully unsupervised model is outperforming conventional methods, such as PCA whereas a semi-supervised method improves upon the unsupervised model results even further.

This code is made publicly available together with the article 
"Dynamic _beta_ VAEs  for quantifying biodiversity by clustering optically recorded insect signals", accepted for publication by Ecological Informatics, 2021-10-05.

This repository provides a minimum working example of the code. As the insect
signals used in the published work are used commercially by FaunaPhotonics,
they are not included in this repository. Instead, a framework for generating
synthesized signals is provided.

# Abstract
While insects are the largest and most diverse group of terrestrial animals,
constituting ca. 80% of all known species, they are difficult to study due
to their small size and similarity between species. Conventional monitor-
ing techniques depend on time consuming trapping methods and tedious
microscope-based work by skilled experts in order to identify the caught in-
sect specimen at species, or even family level. Researchers and policy makers
are in urgent need of a scalable monitoring tool in order to conserve biodi-
versity and secure human food production due to the rapid decline in insect
numbers.

Novel automated optical monitoring equipment can record tens of thou-
sands of insect observations in a single day and the ability to identify key
targets at species level can be a vital tool for entomologists, biologists and
agronomists. Recent work has aimed for a broader analysis using unsuper-
vised clustering as a proxy for conventional biodiversity measures, such as
species richness and species evenness, without actually identifying the species
of the detected target.

In order to improve upon existing insect clustering methods, we propose
an adaptive variant of the variational autoencoder (VAE) which is capable
of clustering data by phylogenetic groups. The proposed dynamic β-VAE
dynamically adapts the scaling of the reconstruction and regularization loss
terms (β value) yielding useful latent representations of the input data. We
demonstrate the usefulness of the dynamic β-VAE on optically recorded insect 
signals from regions of southern Scandinavia to cluster unlabelled targets
into possible species. We also demonstrate improved clustering performance
in a semi-supervised setting using a small subset of labelled data. 

These experimental results, in both unsupervised- and semi-supervised settings, with
the dynamic β-VAE are promising and, in the near future, can be deployed
to monitor insects and conserve the rapidly declining insect biodiversity.


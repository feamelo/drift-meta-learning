# Drift meta learning

Master's thesis in Computer Science at the University of Brasília ([PPGI](http://ppgi.unb.br)).

The objective is to perform drift detection using meta learning by predicting the performance of a base model (regression or classification) with unknown target.

To Do:


- [x] Offline stage MetaLearner implementation
- [x] Online stage MetaLearner implementation
- [x] Meta model incremental learning
- [x] Statistical meta features implementation
- [x] Clustering meta features implementation
- [x] PSI implementation
- [x] Domain classifier implementation
- [x] Meta/base model hyperparameter tuning
- [x] TimeSeries cross validation usage
- [x] Different base model metrics (meta labels) experiments
- [x] Target delay experiments
- [x] Different base model experiments
- [x] Meta base dimensionality reduction experiments
- [ ] Other drift metrics implementation
- [ ] Other databases experiments
- [ ] Measure elapsed time for each meta feature
- [ ] Meta feature selection experiments
- [x] Baseline
- [ ] Evaluation metrics for meta model vs baseline
- [ ] Drift alert definition
- [ ] Drift alert configuration
- [ ] Code documentation
- [ ] Readme
- [ ] Experiments notebooks adjustments to fit new MetaLearner structure
- [ ] Parameter for defining if metabase and base db should be stored
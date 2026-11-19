# cs72-toxicity-nlp

Will Lehman and Todd Rosenbaum

### Quick Links:
GitHub Repository: https://github.com/WillLehman/cs72-toxicity-nlp 

Dataset: https://huggingface.co/datasets/google/civil_comments 

Interactive Web Demo: https://cs72-toxicity-nlp-ezzd5wwjd4rmyzekuzyytr.streamlit.app/ 

DistilBERT model: https://huggingface.co/will55555/cs72-toxicity-distilbert 

Writeup: [link]

### Reproducibility Instructions: 

Please see our interactive web application here:
https://cs72-toxicity-nlp-ezzd5wwjd4rmyzekuzyytr.streamlit.app

This demonstration allows users to dynamically input comments and evaluate classification on both Logistic Regression and DistilBERT models. Additionally, there is optionality to explore bias of our models by considering how inputted identities influence assigned toxicity scores. Finally, counterfactual analysis can be customized to reveal largest classification inconsistencies across select demographics.

We deliberately made our demonstration dynamic and available for user input. The Civil Comments Dataset was collected between 2015-2017. The English language and online media environment have evolved substantially over the past 10 years. As a result, our dataset and corresponding models may be outdated. The dynamic nature of this demonstrations allows users to test model performance on current language and observe this limitation.

Please see our GitHub repository here:
https://github.com/WillLehman/cs72-toxicity-nlp 

The main code for our project is in notebooks 01-04. You can download and run them in order yourself. Note that doing so may take some time because it will re-train the logistic and DistilBERT models. Code for the web application is in app.py. If you want to use our DistilBERT model without retraining, it is also hosted on Hugging Face. A link to it can be found in the references section. If you want to view our results without rerunning the code, simply open notebooks 01-04 to view the pre-loaded charts and outputs. Test predictions are saved as .csv files in the results folder. 

### Additional References:
Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. Proceedings of the 2019 World Wide Web Conference, 491–500. https://arxiv.org/abs/1903.04561	

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, 4171–4186. https://arxiv.org/abs/1810.04805

Google. (n.d.). Civil Comments [Dataset]. Hugging Face. https://huggingface.co/datasets/google/civil_comments
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108. https://arxiv.org/abs/1910.01108

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., et al. (2020). Transformers: State-of-the-art natural language processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. https://aclanthology.org/2020.emnlp-demos.6/


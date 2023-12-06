#!/bin/bash
# create dirs if not exist
mkdir -p data/adult
mkdir -p data/telco
mkdir -p data/creditcardfraud

# inflate datasets into dirs
unzip -o data/adult.zip -d data/adult
unzip -o data/telco-customer-churn.zip -d data/telco
unzip -o data/creditcardfraud.zip -d data/creditcardfraud

/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/ctr_provider.h>
#include <LightGBM/utils/log.h>

#include <random>

namespace LightGBM {

CTRProvider::CTRProvider(const Config* config, const Dataset* train_data):
max_bin_(config->max_bin),
min_data_in_bin_(config->min_data_in_bin),
use_missing_(config->use_missing),
zero_as_missing_(config->zero_as_missing),
min_data_in_leaf_(config->min_data_in_leaf),
num_threads_(config->num_threads),
random_seed_(config->data_random_seed),
num_folds_(config->num_folds_for_ctr),
num_data_(train_data->num_data_),
num_original_total_features_(train_data->num_total_features_),
max_cat_to_one_hot_(config->max_cat_to_onehot) {}

void CTRProvider::ConstructCTRBinMappers(const std::vector<std::vector<double>>& sample_values,
                                         const std::vector<double>& sample_labels,
                                         const int num_sample_data, 
                                         const std::vector<int>& all_sample_indices,
                                         std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
                                         std::vector<std::vector<int>>& sample_indices,
                                         Dataset* train_data,
                                         std::vector<std::string>& feature_names_from_data_loader) {
    CHECK(num_sample_data == static_cast<int>(sample_labels.size()));
    CHECK(num_sample_data == static_cast<int>(all_sample_indices.size()));
    CHECK(sample_values.size() == bin_mappers.size());

    GetCTRMetaInfo(bin_mappers, num_original_total_features_, sample_labels, all_sample_indices, train_data, feature_names_from_data_loader);

    if(num_cat_features_ >= num_threads_) {
        //parallel by features
        CHECK(static_cast<int>(cat_fids_.size()) == num_cat_features_);
        const size_t old_bin_mappers_size = bin_mappers.size();
        bin_mappers.resize(cat_fids_.size() + old_bin_mappers_size);
        sample_indices.resize(cat_fids_.size() + old_bin_mappers_size);
        #pragma omp parallel for schedule(static) num_threads(num_threads_)
        for(int i = 0; i < num_cat_features_; ++i) {
            int fid = cat_fids_[i];
            bin_mappers[i + old_bin_mappers_size].reset(ConstructCTRBinMapper(sample_values[fid], sample_indices[fid], 
                sample_labels, bin_mappers[fid], num_sample_data, all_sample_indices, fid, fold_ctr_values_[i], ctr_values_[i]));
            sample_indices[i + old_bin_mappers_size] = sample_indices[fid];
        }
    }
    else {
        //parallel by data
        for(int i = 0; i < num_cat_features_; ++i) {
            int fid = cat_fids_[i];
            bin_mappers.emplace_back(ConstructCTRBinMapperParallel(sample_values[fid], sample_indices[fid], 
                sample_labels, bin_mappers[fid], num_sample_data, all_sample_indices, fid, fold_ctr_values_[i], ctr_values_[i]));
            sample_indices.push_back(sample_indices[fid]);
        }
    }
}

void CTRProvider::GetCTRMetaInfo(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers, 
                                 const int num_total_features,
                                 const std::vector<double>& sample_labels,
                                 const std::vector<int>& all_sample_indices,
                                 Dataset* train_data,
                                 std::vector<std::string>& feature_names_from_data_loader) {
    //get information about the categorical features, build maps between categorical feature id and ctr feature id
    num_cat_features_ = 0;
    for(int fid = 0; fid < static_cast<int>(bin_mappers.size()); ++fid) {
        if(bin_mappers[fid]->bin_type() == BinType::CategoricalBin && bin_mappers[fid]->num_bin() > max_cat_to_one_hot_) {
            const int cur_ctr_fid = num_total_features + num_cat_features_;
            cat_fid_2_ctr_fid_[fid] = cur_ctr_fid;
            ctr_fid_2_cat_fid_[cur_ctr_fid] = fid;
            cat_fids_.push_back(fid);
            cat_fid_2_inner_cat_fid_[fid] = num_cat_features_;
            ++num_cat_features_;
        }
    }

    //calculate ctr prior, which is the average of sampled labels
    double ctr_prior = 0.0;
    const int num_sample_data = static_cast<int>(sample_labels.size());
    #pragma omp parallel for schedule(static) num_threads(num_threads_) reduction(+:ctr_prior) 
    for(int index = 0; index < num_sample_data; ++index) {
        ctr_prior += sample_labels[index];
    }
    ctr_prior /= num_sample_data;

    //build ctr function
    ctr_function_ = [ctr_prior] (double label_sum, int count) {
        return (label_sum + ctr_prior) / (count + 1.0);
    };

    //prepare ctr value vectors
    fold_ctr_values_.resize(num_cat_features_);
    ctr_values_.resize(num_cat_features_);
    for(int i = 0; i < num_cat_features_; ++i) {
        fold_ctr_values_[i].resize(num_folds_);
        ctr_values_[i].resize(bin_mappers[cat_fids_[i]]->num_bin());
        for(int fold_id = 0; fold_id < num_folds_; ++fold_id) {
            fold_ctr_values_[i][fold_id].resize(bin_mappers[cat_fids_[i]]->num_bin());
        }
    }

    //generate fold partition 
    GenRandomFoldPartition(num_sample_data, all_sample_indices);

    //change dataset meta information
    train_data->num_total_features_ += num_cat_features_;
    train_data->used_feature_map_.resize(train_data->num_total_features_, -1);
    for(int i = 0; i < num_cat_features_; ++i) {
        int cat_fid = cat_fids_[i];
        std::stringstream str_buf;
        str_buf << "@CTR_for_feature_" << cat_fid;
        feature_names_from_data_loader.push_back(str_buf.str());
    }
    train_data->set_feature_names(feature_names_from_data_loader);
}

void CTRProvider::GenRandomFoldPartition(const int num_sample_data, const std::vector<int>& all_sample_indices) {
    CHECK(num_sample_data == static_cast<int>(all_sample_indices.size()));
    std::vector<std::mt19937> mt_generators;
    for(int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        mt_generators.emplace_back(random_seed_ + thread_id);
    }
    const std::vector<double> fold_probs(num_folds_, 1.0 / num_folds_);
    std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
    
    fold_ids_.clear();
    fold_ids_.resize(num_data_, -1);
    
    int block_size = (num_data_ + num_threads_ - 1) / num_threads_;
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
    for(int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        int block_start = block_size * thread_id;
        int block_end = std::min(block_start + block_size, num_data_);
        std::mt19937& generator = mt_generators[thread_id];
        for(int index = block_start; index < block_end; ++index) {
            fold_ids_[index] = fold_distribution(generator);
        }
    }
}

BinMapper* CTRProvider::ConstructCTRBinMapper(const std::vector<double>& sample_values_one_column,
                                              const std::vector<int>& sample_indices_one_column,
                                              const std::vector<double>& sample_labels,
                                              const std::unique_ptr<BinMapper>& bin_mapper,
                                              const int num_sample_data,
                                              const std::vector<int>& all_sample_indices,
                                              const int real_cat_fid,
                                              std::vector<std::vector<double>>& out_fold_ctr_values,
                                              std::vector<double>& out_ctr_values) {
    //calculate ctr 
    const int num_bin = bin_mapper->num_bin();
    std::vector<int> count_per_bin(num_bin * num_folds_, 0);
    std::vector<double> label_sum_per_bin(num_bin * num_folds_, 0);
    const int num_sample_data_one_column = static_cast<int>(sample_values_one_column.size());

    for(int i = 0; i < num_sample_data_one_column; ++i) {
        const double value = sample_values_one_column[i];
        const uint32_t bin = bin_mapper->ValueToBin(value);
        const int index = sample_indices_one_column[i];
        const int real_index = all_sample_indices[index];
        const int fold = fold_ids_[real_index];
        const double label = sample_labels[index];
        const int fold_bin = fold * num_bin + bin;
        ++count_per_bin[fold_bin];
        label_sum_per_bin[fold_bin] += label;   
    }
    
    for(int bin = 0; bin < num_bin; ++bin) {
        int count_per_bin_all_fold = 0;
        double label_sum_per_bin_all_fold = 0.0;
        for(int fold_id = 0; fold_id < num_folds_; ++fold_id) {
            const int fold_bin = fold_id * num_bin + bin;
            count_per_bin_all_fold += count_per_bin[fold_bin];
            label_sum_per_bin_all_fold += label_sum_per_bin[fold_bin];
        }
        out_ctr_values[bin] = ctr_function_(label_sum_per_bin_all_fold, count_per_bin_all_fold);
        for(int fold_id = 0; fold_id < num_folds_; ++fold_id) {
            const int fold_bin = fold_id * num_bin + bin; 
            out_fold_ctr_values[fold_id][bin] = ctr_function_(label_sum_per_bin_all_fold - label_sum_per_bin[fold_bin],
                                                              count_per_bin_all_fold - count_per_bin[fold_bin]);
        }
    }

    //find bin
    std::vector<double> ctr_values(sample_values_one_column.size(), 0.0);
    for(int i = 0; i < num_sample_data_one_column; ++i) {
        const int index = sample_indices_one_column[i];
        const int real_index = all_sample_indices[index];
        const double value = sample_values_one_column[i];
        if(std::isnan(value) || value < 0.0) {
            CHECK(bin_mapper->missing_type() == MissingType::NaN);
            ctr_values[i] = NaN;
        }
        else {
            const int fold = fold_ids_[real_index];
            const uint32_t bin = bin_mapper->ValueToBin(value);
            ctr_values[i] = out_fold_ctr_values[fold][bin];
            CHECK(ctr_values[i] > 0.0);
        }
    }

    BinMapper* ctr_bin_mapper = new BinMapper(); 
    const data_size_t filter_cnt = static_cast<data_size_t>(
        static_cast<double>(min_data_in_leaf_ * num_sample_data) / num_data_);
    ctr_bin_mapper->FindBin(ctr_values.data(), num_sample_data_one_column, num_sample_data, max_bin_, min_data_in_bin_,
        filter_cnt, BinType::NumericalBin, use_missing_, false, true); 
    ctr_bin_mapper->set_ctr_info(real_cat_fid, out_ctr_values, bin_mapper->GetSeenCategories());

    CHECK(ctr_bin_mapper->missing_type() == bin_mapper->missing_type());

    return ctr_bin_mapper; 
}

BinMapper* CTRProvider::ConstructCTRBinMapperParallel(const std::vector<double>& sample_values_one_column,
                            const std::vector<int>& sample_indices_one_column,
                            const std::vector<double>& sample_labels,
                            const std::unique_ptr<BinMapper>& bin_mapper,
                            const int num_sample_data,
                            const std::vector<int>& all_sample_indices,
                            const int real_cat_fid,
                            std::vector<std::vector<double>>& out_fold_ctr_values,
                            std::vector<double>& out_ctr_values) {
    //calculate ctr 
    const int num_bin = bin_mapper->num_bin();
    std::vector<int> thread_count_per_bin(num_bin * num_folds_ * num_threads_, 0);
    std::vector<double> thread_label_sum_per_bin(num_bin * num_folds_ * num_threads_, 0);
    const int num_sample_data_one_column = static_cast<int>(sample_values_one_column.size());
    const int thread_vector_size = num_folds_ * num_bin;

    //accumulate by thread
    int block_size = (num_sample_data_one_column + num_threads_ - 1) / num_threads_;
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
    for(int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        int start = block_size * thread_id;
        int end = std::min(start + block_size, num_sample_data_one_column);
        for(int i = start; i < end; ++i) {
            const double value = sample_values_one_column[i];
            const uint32_t bin = bin_mapper->ValueToBin(value);
            const int index = sample_indices_one_column[i];
            const int real_index = all_sample_indices[index];
            const int fold = fold_ids_[real_index];
            const double label = sample_labels[index];
            const int fold_bin = fold * num_bin + bin;
            const int thread_fold_bin = thread_id * thread_vector_size + fold_bin;
            ++thread_count_per_bin[thread_fold_bin];
            thread_label_sum_per_bin[thread_fold_bin] += label;   
        }
    }

    //sum up, store in the vector of thread 0
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for(int fold_bin = 0; fold_bin < num_bin * num_folds_; ++fold_bin) {
        for(int thread_id = 1; thread_id < num_threads_; ++thread_id) {
            const int thread_fold_bin = thread_id * thread_vector_size + fold_bin;
            thread_count_per_bin[fold_bin] += thread_count_per_bin[thread_fold_bin];
            thread_label_sum_per_bin[fold_bin] += thread_label_sum_per_bin[thread_fold_bin];
        }
    }
    
    //calculate ctr per fold
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for(int bin = 0; bin < num_bin; ++bin) {
        int count_per_bin_all_fold = 0;
        double label_sum_per_bin_all_fold = 0.0;
        for(int fold_id = 0; fold_id < num_folds_; ++fold_id) {
            const int fold_bin = fold_id * num_bin + bin;
            count_per_bin_all_fold += thread_count_per_bin[fold_bin];
            label_sum_per_bin_all_fold += thread_label_sum_per_bin[fold_bin];
        }
        out_ctr_values[bin] = ctr_function_(label_sum_per_bin_all_fold, count_per_bin_all_fold);
        for(int fold_id = 0; fold_id < num_folds_; ++fold_id) {
            const int fold_bin = fold_id * num_bin + bin; 
            out_fold_ctr_values[fold_id][bin] = ctr_function_(label_sum_per_bin_all_fold - thread_label_sum_per_bin[fold_bin],
                                                              count_per_bin_all_fold - thread_count_per_bin[fold_bin]);
        }
    }

    //find bin
    std::vector<double> ctr_values(sample_values_one_column.size(), 0.0);
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for(int i = 0; i < num_sample_data_one_column; ++i) {
        const int index = sample_indices_one_column[i];
        const int real_index = all_sample_indices[index];
        const double value = sample_values_one_column[i];
        if(std::isnan(value) || value < 0.0) {
            CHECK(bin_mapper->missing_type() == MissingType::NaN);
            ctr_values[i] = NaN;
        }
        else {
            const int fold = fold_ids_[real_index];
            const uint32_t bin = bin_mapper->ValueToBin(value);
            ctr_values[i] = out_fold_ctr_values[fold][bin];
            CHECK(ctr_values[i] > 0.0);
        }
    }

    BinMapper* ctr_bin_mapper = new BinMapper(); 
    const data_size_t filter_cnt = static_cast<data_size_t>(
        static_cast<double>(min_data_in_leaf_ * num_sample_data) / num_data_);

    //TODO: optmize FindBin for parallelism
    ctr_bin_mapper->FindBin(ctr_values.data(), num_sample_data_one_column, num_sample_data, max_bin_, min_data_in_bin_,
        filter_cnt, BinType::NumericalBin, use_missing_, false, true); 
    ctr_bin_mapper->set_ctr_info(real_cat_fid, ctr_values, bin_mapper->GetSeenCategories());

    CHECK(ctr_bin_mapper->missing_type() == bin_mapper->missing_type());

    return ctr_bin_mapper;   
}

} // namespace LightGBM
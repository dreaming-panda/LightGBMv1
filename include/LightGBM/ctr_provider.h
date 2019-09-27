/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_CTR_PROVIDER_H_
#define LIGHTGBM_CTR_PROVIDER_H_

#include <vector>
#include <unordered_map>
#include <LightGBM/bin.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>

namespace LightGBM {

class CTRProvider {
public:
    CTRProvider(const Config* config, const Dataset* train_data);

    void ConstructCTRBinMappers(const std::vector<std::vector<double>>& sample_values,
                                const std::vector<double>& sample_labels,
                                const int num_sample_data,
                                const std::vector<int>& all_sample_indices,
                                std::vector<std::unique_ptr<BinMapper>>& bin_mappers, 
                                std::vector<std::vector<int>>& sample_indices,
                                Dataset* train_data,
                                std::vector<std::string>& feature_names_from_data_loader);

    void PushCTRData(const int cat_fid, const int tid, const int real_index, const double value, Dataset* data, bool is_valid) const;

private:
    BinMapper* ConstructCTRBinMapper(const std::vector<double>& sample_values_one_column,
                               const std::vector<int>& sample_indices_one_column,
                               const std::vector<double>& sample_labels,
                               const std::unique_ptr<BinMapper>& bin_mapper,
                               const int num_sample_data,
                               const std::vector<int>& all_sample_indices,
                               const int real_cat_fid,
                               std::vector<std::vector<double>>& out_fold_ctr_values,
                               std::vector<double>& out_ctr_values);

    BinMapper* ConstructCTRBinMapperParallel(const std::vector<double>& sample_values_one_column,
                               const std::vector<int>& sample_indices_one_column,
                               const std::vector<double>& sample_labels,
                               const std::unique_ptr<BinMapper>& bin_mapper,
                               const int num_sample_data,
                               const std::vector<int>& all_sample_indices,
                               const int real_cat_fid,
                               std::vector<std::vector<double>>& out_fold_ctr_values,
                               std::vector<double>& out_ctr_values);

    void GetCTRMetaInfo(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers, 
                        const int num_total_features,
                        const std::vector<double>& sample_labels,
                        const std::vector<int>& all_sample_indices,
                        Dataset* train_data,
                        std::vector<std::string>& feature_names_from_data_loader);

    void GenRandomFoldPartition(const int num_sample_data, const std::vector<int>& all_sample_indices);

    const int max_bin_;
    const int min_data_in_bin_;
    const bool use_missing_;
    const bool zero_as_missing_;
    const int min_data_in_leaf_;
    const int num_threads_;
    const int random_seed_;
    const int num_folds_; 
    const int num_data_;
    const int num_original_total_features_;
    const int max_cat_to_one_hot_;

    std::function<double(double, int)> ctr_function_;
    int num_cat_features_;
    std::vector<int> cat_fids_;
    std::unordered_map<int, int> cat_fid_2_ctr_fid_;
    std::unordered_map<int, int> ctr_fid_2_cat_fid_;
    std::unordered_map<int, int> cat_fid_2_inner_cat_fid_;
    std::vector<std::vector<std::vector<double>>> fold_ctr_values_; 
    std::vector<std::vector<double>> ctr_values_;
    std::vector<int> fold_ids_;
};

inline void CTRProvider::PushCTRData(const int cat_fid, const int tid, const int real_index, const double value, Dataset* data, bool is_valid) const {
    if(cat_fid_2_ctr_fid_.count(cat_fid)) {
        if(!is_valid) {
            const BinMapper* cat_bin_mapper = data->FeatureBinMapper(data->InnerFeatureIndex(cat_fid));
            CHECK(cat_bin_mapper->bin_type() == BinType::CategoricalBin);
            const bool value_seen_in_train = cat_bin_mapper->HasValueInCat(value);
            CHECK(value_seen_in_train == true);
            const int cat_value = static_cast<int>(cat_bin_mapper->ValueToBin(value));
            const int fold_id = fold_ids_[real_index];
            const int inner_cat_fid = cat_fid_2_inner_cat_fid_.at(cat_fid);
            double ctr_value = fold_ctr_values_[inner_cat_fid][fold_id][cat_value];

            const int ctr_fid = cat_fid_2_ctr_fid_.at(cat_fid);
            const int inner_ctr_fid = data->used_feature_map_[ctr_fid];
            const BinMapper* ctr_bin_mapper = data->FeatureBinMapper(inner_ctr_fid);
            CHECK(ctr_bin_mapper->missing_type() == cat_bin_mapper->missing_type());
            if(cat_bin_mapper->missing_type() == MissingType::NaN) {
                if(cat_value == cat_bin_mapper->num_bin() - 1) {
                    ctr_value = NaN;
                }
            }

            if(inner_ctr_fid >= 0) {
                const int group_id = data->feature2group_[inner_ctr_fid];
                const int sub_feature_id = data->feature2subfeature_[inner_ctr_fid];
                data->PushOneData(tid, static_cast<data_size_t>(real_index), group_id, sub_feature_id, ctr_value);
            }
        } 
    }
}

} //namespace LightGBM

#endif //LightGBM_CTR_PROVIDER_H_
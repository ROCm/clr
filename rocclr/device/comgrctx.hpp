/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include <mutex>
#if defined(USE_COMGR_LIBRARY)
#include "top.hpp"
#include "amd_comgr.h"

namespace amd {
typedef void (*t_amd_comgr_get_version)(size_t *major, size_t *minor);
typedef amd_comgr_status_t (*t_amd_comgr_status_string)(amd_comgr_status_t status, const char ** status_string);
typedef amd_comgr_status_t (*t_amd_comgr_get_isa_count)(size_t *count);
typedef amd_comgr_status_t (*t_amd_comgr_get_isa_name)(size_t index, const char **isa_name);
typedef amd_comgr_status_t (*t_amd_comgr_get_isa_metadata)(const char *isa_name, amd_comgr_metadata_node_t *metadata);
typedef amd_comgr_status_t (*t_amd_comgr_create_data)(amd_comgr_data_kind_t kind, amd_comgr_data_t *data);
typedef amd_comgr_status_t (*t_amd_comgr_release_data)(amd_comgr_data_t data);
typedef amd_comgr_status_t (*t_amd_comgr_get_data_kind)(amd_comgr_data_t data, amd_comgr_data_kind_t *kind);
typedef amd_comgr_status_t (*t_amd_comgr_set_data)(amd_comgr_data_t data, size_t size, const char* bytes);
typedef amd_comgr_status_t (*t_amd_comgr_set_data_name)(amd_comgr_data_t data, const char* name);
typedef amd_comgr_status_t (*t_amd_comgr_get_data)(amd_comgr_data_t data, size_t *size, char *bytes);
typedef amd_comgr_status_t (*t_amd_comgr_get_data_name)(amd_comgr_data_t data, size_t *size, char *name);
typedef amd_comgr_status_t (*t_amd_comgr_get_data_isa_name)(amd_comgr_data_t data, size_t *size, char *isa_name);
typedef amd_comgr_status_t (*t_amd_comgr_get_data_metadata)(amd_comgr_data_t data, amd_comgr_metadata_node_t *metadata);
typedef amd_comgr_status_t (*t_amd_comgr_destroy_metadata)(amd_comgr_metadata_node_t metadata);
typedef amd_comgr_status_t (*t_amd_comgr_create_data_set)(amd_comgr_data_set_t *data_set);
typedef amd_comgr_status_t (*t_amd_comgr_destroy_data_set)(amd_comgr_data_set_t data_set);
typedef amd_comgr_status_t (*t_amd_comgr_data_set_add)(amd_comgr_data_set_t data_set, amd_comgr_data_t data);
typedef amd_comgr_status_t (*t_amd_comgr_data_set_remove)(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind);
typedef amd_comgr_status_t (*t_amd_comgr_action_data_count)(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t *count);
typedef amd_comgr_status_t (*t_amd_comgr_action_data_get_data)(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t index, amd_comgr_data_t *data);
typedef amd_comgr_status_t (*t_amd_comgr_create_action_info)(amd_comgr_action_info_t *action_info);
typedef amd_comgr_status_t (*t_amd_comgr_destroy_action_info)(amd_comgr_action_info_t action_info);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_isa_name)(amd_comgr_action_info_t action_info, const char *isa_name);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_get_isa_name)(amd_comgr_action_info_t action_info, size_t *size, char *isa_name);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_language)(amd_comgr_action_info_t action_info, amd_comgr_language_t language);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_get_language)(amd_comgr_action_info_t action_info, amd_comgr_language_t *language);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_option_list)(amd_comgr_action_info_t action_info, const char *options[], size_t count);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_get_option_list_count)(amd_comgr_action_info_t action_info, size_t *count);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_get_option_list_item)(amd_comgr_action_info_t action_info, size_t index, size_t *size, char *option);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_working_directory_path)(amd_comgr_action_info_t action_info, const char *path);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_get_working_directory_path)(amd_comgr_action_info_t action_info, size_t *size, char *path);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_logging)(amd_comgr_action_info_t action_info, bool logging);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_get_logging)(amd_comgr_action_info_t action_info, bool *logging);
typedef amd_comgr_status_t (*t_amd_comgr_do_action)(amd_comgr_action_kind_t kind, amd_comgr_action_info_t info, amd_comgr_data_set_t input, amd_comgr_data_set_t result);
typedef amd_comgr_status_t (*t_amd_comgr_get_metadata_kind)(amd_comgr_metadata_node_t metadata, amd_comgr_metadata_kind_t *kind);
typedef amd_comgr_status_t (*t_amd_comgr_get_metadata_string)(amd_comgr_metadata_node_t metadata, size_t *size, char *string);
typedef amd_comgr_status_t (*t_amd_comgr_get_metadata_map_size)(amd_comgr_metadata_node_t metadata, size_t *size);
typedef amd_comgr_status_t (*t_amd_comgr_iterate_map_metadata)(amd_comgr_metadata_node_t metadata, amd_comgr_status_t(*callback)(amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void *user_data), void *user_data);
typedef amd_comgr_status_t (*t_amd_comgr_metadata_lookup)(amd_comgr_metadata_node_t metadata, const char *key, amd_comgr_metadata_node_t *value);
typedef amd_comgr_status_t (*t_amd_comgr_get_metadata_list_size)(amd_comgr_metadata_node_t metadata, size_t *size);
typedef amd_comgr_status_t (*t_amd_comgr_index_list_metadata)(amd_comgr_metadata_node_t metadata, size_t index, amd_comgr_metadata_node_t *value);
typedef amd_comgr_status_t (*t_amd_comgr_iterate_symbols)(amd_comgr_data_t data, amd_comgr_status_t(*callback)(amd_comgr_symbol_t symbol, void *user_data), void *user_data);
typedef amd_comgr_status_t (*t_amd_comgr_symbol_lookup)(amd_comgr_data_t data, const char *name, amd_comgr_symbol_t *symbol);
typedef amd_comgr_status_t (*t_amd_comgr_symbol_get_info)(amd_comgr_symbol_t symbol, amd_comgr_symbol_info_t attribute, void *value);

struct ComgrEntryPoints {
  void* handle;
  t_amd_comgr_get_version               amd_comgr_get_version;
  t_amd_comgr_status_string             amd_comgr_status_string;
  t_amd_comgr_get_isa_count             amd_comgr_get_isa_count;
  t_amd_comgr_get_isa_name              amd_comgr_get_isa_name;
  t_amd_comgr_get_isa_metadata          amd_comgr_get_isa_metadata;
  t_amd_comgr_create_data               amd_comgr_create_data;
  t_amd_comgr_release_data              amd_comgr_release_data;
  t_amd_comgr_get_data_kind             amd_comgr_get_data_kind;
  t_amd_comgr_set_data                  amd_comgr_set_data;
  t_amd_comgr_set_data_name             amd_comgr_set_data_name;
  t_amd_comgr_get_data                  amd_comgr_get_data;
  t_amd_comgr_get_data_name             amd_comgr_get_data_name;
  t_amd_comgr_get_data_isa_name         amd_comgr_get_data_isa_name;
  t_amd_comgr_get_data_metadata         amd_comgr_get_data_metadata;
  t_amd_comgr_destroy_metadata          amd_comgr_destroy_metadata;
  t_amd_comgr_create_data_set           amd_comgr_create_data_set;
  t_amd_comgr_destroy_data_set          amd_comgr_destroy_data_set;
  t_amd_comgr_data_set_add              amd_comgr_data_set_add;
  t_amd_comgr_data_set_remove           amd_comgr_data_set_remove;
  t_amd_comgr_action_data_count         amd_comgr_action_data_count;
  t_amd_comgr_action_data_get_data      amd_comgr_action_data_get_data;
  t_amd_comgr_create_action_info        amd_comgr_create_action_info;
  t_amd_comgr_destroy_action_info       amd_comgr_destroy_action_info;
  t_amd_comgr_action_info_set_isa_name  amd_comgr_action_info_set_isa_name;
  t_amd_comgr_action_info_get_isa_name  amd_comgr_action_info_get_isa_name;
  t_amd_comgr_action_info_set_language  amd_comgr_action_info_set_language;
  t_amd_comgr_action_info_get_language  amd_comgr_action_info_get_language;
  t_amd_comgr_action_info_set_option_list   amd_comgr_action_info_set_option_list;
  t_amd_comgr_action_info_get_option_list_count   amd_comgr_action_info_get_option_list_count;
  t_amd_comgr_action_info_get_option_list_item   amd_comgr_action_info_get_option_list_item;
  t_amd_comgr_action_info_set_working_directory_path  amd_comgr_action_info_set_working_directory_path;
  t_amd_comgr_action_info_get_working_directory_path  amd_comgr_action_info_get_working_directory_path;
  t_amd_comgr_action_info_set_logging   amd_comgr_action_info_set_logging;
  t_amd_comgr_action_info_get_logging   amd_comgr_action_info_get_logging;
  t_amd_comgr_do_action                 amd_comgr_do_action;
  t_amd_comgr_get_metadata_kind         amd_comgr_get_metadata_kind;
  t_amd_comgr_get_metadata_string       amd_comgr_get_metadata_string;
  t_amd_comgr_get_metadata_map_size     amd_comgr_get_metadata_map_size;
  t_amd_comgr_iterate_map_metadata      amd_comgr_iterate_map_metadata;
  t_amd_comgr_metadata_lookup           amd_comgr_metadata_lookup;
  t_amd_comgr_get_metadata_list_size    amd_comgr_get_metadata_list_size;
  t_amd_comgr_index_list_metadata       amd_comgr_index_list_metadata;
  t_amd_comgr_iterate_symbols           amd_comgr_iterate_symbols;
  t_amd_comgr_symbol_lookup             amd_comgr_symbol_lookup;
  t_amd_comgr_symbol_get_info           amd_comgr_symbol_get_info;
};

#ifdef COMGR_DYN_DLL 
#define DYN(NAME) cep_.NAME
#define GET_COMGR_SYMBOL(NAME) cep_.NAME = \
  reinterpret_cast<t_##NAME>(Os::getSymbol(cep_.handle, #NAME)); \
  if (nullptr == cep_.NAME) { return false; }
#else
#define DYN(NAME) NAME
#define GET_COMGR_SYMBOL(NAME)
#endif

class Comgr : public amd::AllStatic {
public:
  static std::once_flag initialized;

  static bool LoadLib();

  static bool IsReady() { return is_ready_; }

  static void get_version(size_t *major, size_t *minor) {
    DYN(amd_comgr_get_version)(major, minor);
  }
  static amd_comgr_status_t status_string(amd_comgr_status_t status, const char ** status_string) {
    return DYN(amd_comgr_status_string)(status, status_string);
  }
  static amd_comgr_status_t get_isa_count(size_t *count) {
    return DYN(amd_comgr_get_isa_count)(count);
  }
  static amd_comgr_status_t get_isa_name(size_t index, const char **isa_name) {
    return DYN(amd_comgr_get_isa_name)(index, isa_name);
  }
  static amd_comgr_status_t get_isa_metadata(const char *isa_name, amd_comgr_metadata_node_t *metadata) {
    return DYN(amd_comgr_get_isa_metadata)(isa_name, metadata);
  }
  static amd_comgr_status_t create_data(amd_comgr_data_kind_t kind, amd_comgr_data_t *data) {
    return DYN(amd_comgr_create_data)(kind, data);
  }
  static amd_comgr_status_t release_data(amd_comgr_data_t data) {
    return DYN(amd_comgr_release_data)(data);
  }
  static amd_comgr_status_t get_data_kind(amd_comgr_data_t data, amd_comgr_data_kind_t *kind) {
    return DYN(amd_comgr_get_data_kind)(data, kind);
  }
  static amd_comgr_status_t set_data(amd_comgr_data_t data, size_t size, const char* bytes) {
    return DYN(amd_comgr_set_data)(data, size, bytes);
  }
  static amd_comgr_status_t set_data_name(amd_comgr_data_t data, const char* name) {
    return DYN(amd_comgr_set_data_name)(data, name);
  }
  static amd_comgr_status_t get_data(amd_comgr_data_t data, size_t *size, char *bytes) {
    return DYN(amd_comgr_get_data)(data, size, bytes);
  }
  static amd_comgr_status_t get_data_name(amd_comgr_data_t data, size_t *size, char *name) {
    return DYN(amd_comgr_get_data_name)(data, size, name);
  }
  static amd_comgr_status_t get_data_isa_name(amd_comgr_data_t data, size_t *size, char *isa_name) {
    return DYN(amd_comgr_get_data_isa_name)(data, size, isa_name);
  }
  static amd_comgr_status_t get_data_metadata(amd_comgr_data_t data, amd_comgr_metadata_node_t *metadata) {
    return DYN(amd_comgr_get_data_metadata)(data, metadata);
  }
  static amd_comgr_status_t destroy_metadata(amd_comgr_metadata_node_t metadata) {
    return DYN(amd_comgr_destroy_metadata)(metadata);
  }
  static amd_comgr_status_t create_data_set(amd_comgr_data_set_t *data_set) {
    return DYN(amd_comgr_create_data_set)(data_set);
  }
  static amd_comgr_status_t destroy_data_set(amd_comgr_data_set_t data_set) {
    return DYN(amd_comgr_destroy_data_set)(data_set);
  }
  static amd_comgr_status_t data_set_add(amd_comgr_data_set_t data_set, amd_comgr_data_t data) {
    return DYN(amd_comgr_data_set_add)(data_set, data);
  }
  static amd_comgr_status_t data_set_remove(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind) {
    return DYN(amd_comgr_data_set_remove)(data_set, data_kind);
  }
  static amd_comgr_status_t action_data_count(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t *count) {
    return DYN(amd_comgr_action_data_count)(data_set, data_kind, count);
  }
  static amd_comgr_status_t action_data_get_data(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t index, amd_comgr_data_t *data) {
    return DYN(amd_comgr_action_data_get_data)(data_set, data_kind, index, data);
  }
  static amd_comgr_status_t create_action_info(amd_comgr_action_info_t *action_info) {
    return DYN(amd_comgr_create_action_info)(action_info);
  }
  static amd_comgr_status_t destroy_action_info(amd_comgr_action_info_t action_info) {
    return DYN(amd_comgr_destroy_action_info)(action_info);
  }
  static amd_comgr_status_t action_info_set_isa_name(amd_comgr_action_info_t action_info, const char *isa_name) {
    return DYN(amd_comgr_action_info_set_isa_name)(action_info, isa_name);
  }
  static amd_comgr_status_t action_info_get_isa_name(amd_comgr_action_info_t action_info, size_t *size, char *isa_name) {
    return DYN(amd_comgr_action_info_get_isa_name)(action_info, size, isa_name);
  }
  static amd_comgr_status_t action_info_set_language(amd_comgr_action_info_t action_info, amd_comgr_language_t language) {
    return DYN(amd_comgr_action_info_set_language)(action_info, language);
  }
  static amd_comgr_status_t action_info_get_language(amd_comgr_action_info_t action_info, amd_comgr_language_t *language) {
    return DYN(amd_comgr_action_info_get_language)(action_info, language);
  }
  static amd_comgr_status_t action_info_set_option_list(amd_comgr_action_info_t action_info, const char *options[], size_t count) {
    return DYN(amd_comgr_action_info_set_option_list)(action_info, options, count);
  }
  static amd_comgr_status_t action_info_get_option_list_count(amd_comgr_action_info_t action_info, size_t *count) {
    return DYN(amd_comgr_action_info_get_option_list_count)(action_info, count);
  }
  static amd_comgr_status_t action_info_get_option_list_item(amd_comgr_action_info_t action_info, size_t index, size_t *size, char *option) {
    return DYN(amd_comgr_action_info_get_option_list_item)(action_info, index, size, option);
  }
  static amd_comgr_status_t action_info_set_working_directory_path(amd_comgr_action_info_t action_info, const char *path) {
    return DYN(amd_comgr_action_info_set_working_directory_path)(action_info, path);
  }
  static amd_comgr_status_t action_info_get_working_directory_path(amd_comgr_action_info_t action_info, size_t *size, char *path) {
    return DYN(amd_comgr_action_info_get_working_directory_path)(action_info, size, path);
  }
  static amd_comgr_status_t action_info_set_logging(amd_comgr_action_info_t action_info, bool logging) {
    return DYN(amd_comgr_action_info_set_logging)(action_info, logging);
  }
  static amd_comgr_status_t action_info_get_logging(amd_comgr_action_info_t action_info, bool *logging) {
    return DYN(amd_comgr_action_info_get_logging)(action_info, logging);
  }
  static amd_comgr_status_t do_action(amd_comgr_action_kind_t kind, amd_comgr_action_info_t info, amd_comgr_data_set_t input, amd_comgr_data_set_t result) {
    return DYN(amd_comgr_do_action)(kind, info, input, result);
  }
  static amd_comgr_status_t get_metadata_kind(amd_comgr_metadata_node_t metadata, amd_comgr_metadata_kind_t *kind) {
    return DYN(amd_comgr_get_metadata_kind)(metadata, kind);
  }
  static amd_comgr_status_t get_metadata_string(amd_comgr_metadata_node_t metadata, size_t *size, char *string) {
    return DYN(amd_comgr_get_metadata_string)(metadata, size, string);
  }
  static amd_comgr_status_t get_metadata_map_size(amd_comgr_metadata_node_t metadata, size_t *size) {
    return DYN(amd_comgr_get_metadata_map_size)(metadata, size);
  }
  static amd_comgr_status_t iterate_map_metadata(amd_comgr_metadata_node_t metadata, amd_comgr_status_t(*callback)(amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void *user_data), void *user_data) {
    return DYN(amd_comgr_iterate_map_metadata)(metadata, callback, user_data);
  }
  static amd_comgr_status_t metadata_lookup(amd_comgr_metadata_node_t metadata, const char *key, amd_comgr_metadata_node_t *value) {
    return DYN(amd_comgr_metadata_lookup)(metadata, key, value);
  }
  static amd_comgr_status_t get_metadata_list_size(amd_comgr_metadata_node_t metadata, size_t *size) {
    return DYN(amd_comgr_get_metadata_list_size)(metadata, size);
  }
  static amd_comgr_status_t index_list_metadata(amd_comgr_metadata_node_t metadata, size_t index, amd_comgr_metadata_node_t *value) {
    return DYN(amd_comgr_index_list_metadata)(metadata, index, value);
  }
  static amd_comgr_status_t iterate_symbols(amd_comgr_data_t data, amd_comgr_status_t(*callback)(amd_comgr_symbol_t symbol, void *user_data), void *user_data) {
    return DYN(amd_comgr_iterate_symbols)(data, callback, user_data);
  }
  static amd_comgr_status_t symbol_lookup(amd_comgr_data_t data, const char *name, amd_comgr_symbol_t *symbol) {
    return DYN(amd_comgr_symbol_lookup)(data, name, symbol);
  }
  static amd_comgr_status_t symbol_get_info(amd_comgr_symbol_t symbol, amd_comgr_symbol_info_t attribute, void *value) {
    return DYN(amd_comgr_symbol_get_info)(symbol, attribute, value);
  }

private:
  static ComgrEntryPoints cep_;
  static bool is_ready_;
};

}
#endif

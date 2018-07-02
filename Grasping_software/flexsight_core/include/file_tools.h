#pragma once

#include <vector>
#include <string>
#include <iostream>     
#include <fstream> 
#include <utility>
#include <ctime>

#include <opencv2/opencv.hpp>

bool readFileNames ( const std::string &input_file_name, std::vector<std::string> &names );
bool readFileNames ( const std::string &input_file_name, std::vector< std::vector<std::string> > &names );

void stripExtension( const std::vector< std::string >& names, const std::string ext,
                     std::vector< std::string >& stripped_names );
std::string generateYamlFilename ( const std::string &name );
void readSamples( const std::vector< std::string> &filenames, cv::Mat &samples,
                  std::vector< std::pair<int,int> > &samples_map );

// Generates max random numbers between num numbers without repetitions
void sampleIndexes ( int max, int num, std::vector<int> &output_indices,
                     std::vector<int> &remaining_indices );
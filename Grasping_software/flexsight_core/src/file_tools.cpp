#include "file_tools.h"

#include <iostream>
#include <fstream>
#include <list>
#include <cstdio>
#include <stdexcept>
#include <locale.h>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace boost;

bool readFileNames ( const string& input_file_name, vector< string >& names )
{
  names.clear();
  ifstream f ( input_file_name.c_str() );

  if ( !f )
  {
    cout<<"Error opening file"<<endl;
    return false;
  }

  string line;

  while ( getline ( f, line ) )
  {
    if( !line.empty())
      names.push_back ( line );
  }

  f.close();

  return true;
}

bool readFileNames ( const string& input_file_name, vector< vector< string > >& names )
{
  names.clear();

  ifstream f ( input_file_name.c_str() );
  string s_line;

  if ( !f )
  {
    cout<<"Error opening file"<<endl;
    return false;
  }

  bool first_line = true;
  int num_images_row = 0;

  char_separator<char> sep ( " " );

  string line;

  while ( getline ( f, line ) )
  {
    if( !line.empty())
    {
      names.push_back ( vector<string>() );
      vector<string> &image_names = names.back();

      tokenizer< char_separator<char> > tokens ( line, sep );
      for ( const auto& t : tokens )
      {
        image_names.push_back ( t );
      }

      if ( first_line )
      {
        first_line = false;
        num_images_row = image_names.size();
      }
      else
      {
        if ( int ( image_names.size() ) != num_images_row )
        {
          names.clear();
          cerr<<"Corrupted file"<<endl;
          return false;
        }
      }
    }
  }

  f.close();

  if ( first_line )
  {
    return false;
  }
  else
  {
    return true;
  }
}

void stripExtension( const vector<string> &names, const string ext,
                     vector<string> &stripped_names )
{
  stripped_names.clear();
  stripped_names.reserve(names.size());

  for( auto &s : names )
  {
    size_t found = s.rfind(ext);
    if (found != string::npos)
      stripped_names.push_back(s.substr(0, found));
    else
      stripped_names.push_back(s);
  }
}

string generateYamlFilename ( const string &name )
{
  const string ending1(".yml"), ending2(".yaml");
  locale loc;
  string tmp_name( name );

  for( uint i = 0; i < name.size(); i++)
    tmp_name[i] = tolower( name[i], loc );
  if ( tmp_name.length() >= ending1.length() &&
       tmp_name.compare ( tmp_name.length() - ending1.length(), ending1.length(), ending1 ) == 0 )
    return name;
  else if( tmp_name.length() >= ending2.length() &&
           tmp_name.compare ( tmp_name.length() - ending2.length(), ending2.length(), ending2 ) == 0 )
    return name;
  else
    return name + ending2;
}

void readSamples( const vector< string> &filenames, cv::Mat &samples,
                  vector< pair<int,int> > &samples_map )
{
  setlocale (LC_NUMERIC,"C");

  samples_map.clear();

  string line;
  list<string> lines;
  int samples_size = 0, samples_dim = 0;
  // Read all files, count the samples and fill samples_map
  for ( uint i = 0; i < filenames.size(); i++ )
  {
    ifstream infile;
    infile.open ( filenames[i].c_str() );

    int file_samples = 0;
    if ( infile.is_open() )
    {
      while ( getline ( infile,line ) )
      {
        lines.push_back(line);
        file_samples++;
      }
      samples_map.push_back(pair<int,int>(samples_size,file_samples));
      samples_size += file_samples;
    }
    else
      cout<<"Can't open "<<filenames[i]<<endl;
  }

  stringstream line_stream(lines.front());
  double value;
  while( (line_stream >> value ).good() ) samples_dim++;

  samples = cv::Mat( samples_size, samples_dim, CV_32F );
  cout<<"Samples size : "<<samples_size<<" X "<<samples_dim<<endl;

  int row = 0, col = 0;
  for( list<string>::iterator iter = lines.begin(); iter != lines.end(); iter++ )
  {
    col = 0;
    float *s_ptr = samples.ptr<float>(row++);
    const char *c_line = (*iter).c_str();
    int i = 0, i_start, line_len = strlen(c_line);
    bool look_for_start = true;
    while( i < line_len && col < samples_dim )
    {
      if( look_for_start )
      {
        if( c_line[i] != ' ' )
        {
          look_for_start = false;
          i_start = i;
        }
      }
      else
      {
        if( c_line[i] == ' ' )
        {
          look_for_start = true;
          *s_ptr++ = float(atof(&c_line[i_start]));
          col++;
        }
      }
      i++;
    }

    if( !look_for_start )
    {
      *s_ptr++ = float(atof(&c_line[i_start]));
      col++;
    }

    if( col < samples_dim )
      throw runtime_error("Corrupted files");
    if( !(row % 10000) )
      cout<<"Reading files... "<<int(round(100.0*double(row)/double(samples_size)))<<"% done"<<endl;
  }
  cout<<"Reading files... 100% done"<<endl;

  setlocale (LC_NUMERIC,"");
}

// Generates max random numbers between num numbers without repetitions
void sampleIndexes ( int max, int num, vector<int> &output_indices, vector<int> &remaining_indices )
{
  max++;
  srand ( time ( 0 ) );

  vector<int> value ( max,-1 );
  vector<int> indices ( num,-1 );

  //generate random numbers:
  for ( int i=0; i < num; i++ )
  {
    bool check; //variable to check or number is already used
    int n; //variable to store the number in
    do
    {
      n = rand()%max;
      //check or number is already used:
      check=true;

      if ( value[n] == n )
        check = false;
    }
    while ( !check ); //loop until new, unique number is found
    value[n] = n; //store the generated number in the array
    indices[i] = n;
  }

  output_indices.resize ( num );

  for ( int i=0; i < num; i++ )
    output_indices[i] = value[indices[i]];

  int remaining_size = max - num + 1;
  remaining_indices.clear();
  remaining_indices.reserve( remaining_size );
  for ( int i=0; i < max ; i++ )
    if( value[i] < 0 )
      remaining_indices.push_back(i);
}
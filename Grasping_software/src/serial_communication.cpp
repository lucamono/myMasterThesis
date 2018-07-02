#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <sys/select.h>
#include <sys/time.h>
#include <signal.h>
#include <sys/ioctl.h>

#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

#include "serial_communication.h"

#define COMM_BUFFER_LEN 255
#define COMM_USLEEP 75000
#define COMM_TIMEOUT_MS 1000

namespace serial_communication
{

using namespace std;

static const char *prefix=":DrT:";

static speed_t getCommSpeed( BaudRate br )
{
  switch( br )
  {
    case BAUD_RATE_115200:
      return B115200;
    case BAUD_RATE_57600:
      return B57600;
    case BAUD_RATE_38400:
      return B38400;
    case BAUD_RATE_19200:
      return B19200;
    case BAUD_RATE_9600:
      return B9600;
    case BAUD_RATE_2400:
      return B2400;
    default:
      return B115200;
  }
}

static void setupParameters( struct termios &options, ParameterSetting settings )
{
  switch( settings )
  {
    case PS_8E1:
      /* 8 data bits, even parity, stop and mask bits (8E1) */
      options.c_cflag |= PARENB;
      options.c_cflag &= ~PARODD;
      options.c_cflag &= ~CSTOPB;
      options.c_cflag &= ~CSIZE;
      options.c_cflag |= CS8;
      break;

    case PS_8N1:
    default:
      /* 8 data bits, no parity, stop and mask bits (8N1) */
      options.c_cflag &= ~PARENB;
      options.c_cflag &= ~CSTOPB;
      options.c_cflag &= ~CSIZE;
      options.c_cflag |= CS8;
      break;
  }
}

int openSerialPort( const char *device_name, BaudRate br, ParameterSetting settings )
{
  /* File descriptor for the port */
  int fd = -1;

  if ( (fd = open (device_name, O_RDWR | O_NOCTTY | O_NDELAY ) ) < 0)
  {
    fprintf(stderr,"open_serial_port : Unable to open %s\n",device_name);
    return -1;
  }

  /* Setting blocking behavior for read call */
  fcntl( fd, F_SETFL, 0 );

  /* Turn on exclusive mode. No other open() is permitted on the device. */
  ioctl( fd, TIOCEXCL, NULL );

  /* The termios structure contains all of the serial options */
  struct termios options;

  /* Get the parameters associated with the terminal */
  if( tcgetattr( fd, &options ) < 0)
  {
    fprintf(stderr,"open_serial_port : tcgetattr() failed!\n");
    close( fd );
    return -1;
  }

  /* Set line discipline  */
  options.c_line = N_TTY;  

  speed_t speed = getCommSpeed( br );
  /* Setting baudrate */
  if( cfsetispeed(&options, speed) < 0 ||
      cfsetospeed(&options, speed) < 0 )
  {
    fprintf(stderr,"open_serial_port : cfsetospeed() failed!\n");
    close( fd );
    return -1;
  }

  /* Enable receiver and don't change te owner of the port */
  options.c_cflag |= (CLOCAL | CREAD);
  
  setupParameters( options, settings );
  
  /* Raw input, no echo */
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  /* Ignore parity errors (input option) */
  options.c_iflag = IGNPAR;
  /* Raw output */
  options.c_oflag &= ~OPOST;

  /* Set the new options for the port */
  if( tcsetattr( fd, TCSANOW, &options ) < 0)
  {
    fprintf(stderr,"open_serial_port : tcsetattr() failed!\n");
    close( fd );
    return -1;
  }

  return fd;
}

void closeSerialPort( int fd )
{
  close( fd );
}

int send( int fd, const char *buffer, int n_elem )
{
  int func_status = write(fd,buffer,n_elem);

  if(func_status < 0)
  {
    fprintf(stderr,"send : write() failed!\n");
    return -1;
  }

  return func_status;
}

int receive(int fd, char *buffer, int n_elem )
{
  int func_status = read(fd, (char *)buffer, n_elem);
  if(func_status < 0)
  {
    fprintf(stderr,"receive : read() failed!\n");
    return -1;
  }
  
  return func_status;
}

int receiveTimeout(int fd, char *buffer, int n_elem, unsigned int timeout_ms)
{

  int func_status = -1;

  fd_set input;
  struct timeval timeout;
  char read_frame;

  FD_ZERO(&input);
  FD_SET(fd, &input);

  /* Select() timeout */
  timeout.tv_sec  = timeout_ms/1000;
  timeout.tv_usec = 1000*(timeout_ms%1000);

  /* Do the select */
  int sel_status = select(fd + 1, &input, NULL, NULL, &timeout);

  /* See if there was an error */
  if (sel_status < 0)
  {
    fprintf(stderr,"receive_timeout : select() failed!\n");
    func_status = -1;
  }
  else if (sel_status == 0)
  {
    /* Timeout */
    func_status =  0;
  }
  else
  {
    if( (func_status = read(fd, (char *)buffer, n_elem)) < 0)
    {
      fprintf(stderr,"receive_timeout : read() failed!\n");
      return -1;
    }      
  }

  return func_status;
}


string sendAndReceive(int fd, string to_send)
{
  int len=to_send.size();

  cout<<"sending: "<<to_send.c_str();
  if( send( fd, to_send.c_str(), len) < 0 )
      fprintf(stderr, "%s : send() failed\n", to_send.c_str());
  else
  {
    char comm_buffer_r[COMM_BUFFER_LEN];
    usleep(COMM_USLEEP);
    int len = receiveTimeout( fd, comm_buffer_r, COMM_BUFFER_LEN, COMM_TIMEOUT_MS );
    if( len > 0 )
    {
      printf("answer: %s\n", comm_buffer_r);
      string ans(comm_buffer_r);
      return ans;
    }
    else if( len == 0)
      fprintf(stderr,"receiveTimeout() timeout!\n");
    else
      fprintf(stderr,"receiveTimeout() failed!\n");
  }
  return "";
}


void sendPoseFrame(int fd, const double* pos)
{
  double d_tot=0;
  int i_tot=0;
  for(int i=0; i<6; i++)
  {
    d_tot+=pos[i];

    stringstream ss;
    ss<<prefix<<"RR "<<4001+i<<" "<<pos[i]<<"\r\n";

    string to_send=ss.str();
    sendAndReceive(fd, to_send);
  }

  i_tot=floor(d_tot);

  stringstream ss_1;
  ss_1<<prefix<<"RR 4007 "<<i_tot<<"\r\n";

  string to_send_1=ss_1.str();
  sendAndReceive(fd, to_send_1);
}

void sendHandShaking(int fd, int flag)
{
  stringstream ss;
  ss<<prefix<<"RR 4008 "<<flag<<"\r\n";
  string to_send=ss.str();
  sendAndReceive(fd, to_send);
}

int readHandShaking(int fd)
{
  stringstream ss;
  ss<<prefix<<"D RR 4008 \r\n";
  string to_send=ss.str();
//   usleep(COMM_USLEEP);
  string ans=sendAndReceive(fd, to_send);
  size_t idx=5;
  return stoi (ans.substr(idx));
}


}

#pragma once

namespace serial_communication
{
  
enum BaudRate
{
  BAUD_RATE_115200,
  BAUD_RATE_57600,
  BAUD_RATE_38400,
  BAUD_RATE_19200,
  BAUD_RATE_9600,
  BAUD_RATE_2400
};

enum ParameterSetting
{
  PS_8N1,
  PS_8E1
};

/*! \fn open_connection()
  Open the serial port with name "device_name" with a baudrate defined by br.
  Returns the file descriptor on success or -1 on error */
int openSerialPort( const char *device_name, BaudRate br,  ParameterSetting settings );

/*! \fn close_serial_port()
  Close the serial port with the file descriptor fd.
*/
void closeSerialPort( int fd );

/*! \fn send()
  Write from the array pointed to by buffer n_elem bytes 
  to the serial port with the file descriptor fd.
  Returns the number of bytes sent or -1 if an error occurred */
int send( int fd, const char *buffer, int n_elem );

/*! \fn receive()
  Read to the the array pointed to by buffer up to n_elem bytes 
  from the serial port with the file descriptor fd.
  Returns the number of bytes sent or -1 if an error occurred */
int receive(int fd, char *buffer, int n_elem );

/*! \fn receive_timeout()
  Read to the the array pointed to by buffer up to n_elem bytes 
  from the serial port with the file descriptor fd.
  Returns the number of bytes sent, 0 if the timeout period 
  timeout_ms (in milliseconds) elapsed prior to receive something 
  or -1 if an error occurred */
int receiveTimeout(int fd, char *buffer, int n_elem, unsigned int timeout_ms);


/*** Higher level functions ***/

std::string sendAndReceive(int fd, std::string to_send);
void sendPoseFrame(int fd, const double* pos);
void sendHandShaking(int fd, int flag);
int readHandShaking(int fd);

};


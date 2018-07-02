# TO BE SET PROPERLY
#set(FREENECT2_INCLUDE_DIRS /home/marco/freenect2/include)
set(FREENECT2_INCLUDE_DIRS /home/luca/libfreenect2/install/include)

# TO BE SET PROPERLY 
#find_library(FREENECT2_LIBRARY NAMES freenect2
#  HINTS
#  /home/marco/freenect2/lib/
#  /home/marco/freenect2/
#)
find_library(FREENECT2_LIBRARY NAMES freenect2
  HINTS
  /home/luca/libfreenect2/install/lib
  /home/luca/libfreenect2/install/include
)

if(FREENECT2_INCLUDE_DIRS AND FREENECT2_LIBRARY)
  set(FREENECT2_FOUND TRUE)
endif()

if(FREENECT2_LIBRARY)
    set(FREENECT2_LIBRARY ${FREENECT2_LIBRARY})
endif()

if (FREENECT2_FOUND)
  MESSAGE("-- Found Freenect2 ${FREENECT2_LIBRARY}")
  #mark_as_advanced(FREENECT2_INCLUDE_DIRS FREENECT2_LIBRARY FREENECT2_LIBRARIES)
endif()

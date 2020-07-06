#!/bin/awk -f
#
# Print memory usage
#
# STM32F098

BEGIN { print "************************************************************"
  MCU="STM32F098";
  print "MCU: ", MCU, PROFILE
  print ""
}
{
  if ($1 == ".text") {
    SIZE = 256 * 1024;
    print ".text:   ", $2, "/", SIZE, "=", 100*$2/SIZE, "%"
  }
  if ($1 == ".rodata") {
    SIZE = 256 * 1024;
    print ".rodata: ", $2, "/", SIZE, "=", 100*$2/SIZE, "%"
    print ""
  }
}
END { print "************************************************************" }

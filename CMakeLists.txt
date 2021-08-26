cmake_minimum_required(VERSION 3.13.4)

project(SliceIntersection)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SliceIntersection")
set(EXTENSION_CATEGORY "Slice")
set(EXTENSION_CONTRIBUTORS "Parry Choi (HKU)")
set(EXTENSION_DESCRIPTION "Slice Intersection Interaction")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/SliceIntersection.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/SliceIntersection/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(IntersectionControls)
add_subdirectory(Philips4dUsDicomPatcher)
add_subdirectory(PhilipsDicomPatcher)
add_subdirectory(SliceFlipping)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

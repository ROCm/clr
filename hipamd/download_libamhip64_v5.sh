#!/bin/bash
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

function download_and_extract_deb()
{
    # Download 5.7.1 hip runtime deb package
    wget -O hip-runtime-amd.deb $1

    # Extract libamdhip64.so.5.7.50701
    dpkg-deb --fsys-tarfile hip-runtime-amd.deb | tar xf - ./opt/rocm-5.7.1/lib/libamdhip64.so.5.7.50701

    # Rename to libamdhip64.so.5
    mv ./opt/rocm-5.7.1/lib/libamdhip64.so.5.7.50701 libamdhip64.so.5

    # Clean up
    rm -r ./opt; rm hip-runtime-amd.deb
}

function download_and_extract_rpm()
{
    # Download 5.7.1 hip runtime rpm package
    wget -O hip-runtime-amd.rpm $1

    # Extract libamdhip64.so.5.7.50701
    rpm2cpio hip-runtime-amd.rpm | cpio -idm

    # Rename to libamdhip64.so.5
    mv ./opt/rocm-5.7.1/lib/libamdhip64.so.5.7.50701 libamdhip64.so.5

    # Clean up
    rm -r ./opt; rm hip-runtime-amd.rpm
}

# Detect OS
ID=$(sed -n 's/^ID=//p' /etc/os-release)
VERSION_ID=$(sed -n 's/^VERSION_ID=//p' /etc/os-release)
PLATFORM_ID=$(sed -n 's/^PLATFORM_ID=//p' /etc/os-release)
echo "ID=$ID"
echo "VERSION_ID=$VERSION_ID"
echo "PLATFORM_ID=$PLATFORM_ID"

# Download and extract libamdhip64.so.5.7.50701 based on OS
case "$ID" in
    ubuntu)
        # Ubuntu
        if [ "$VERSION_ID" == "\"20.04\"" ]; then
            # Ubuntu 20.04
            download_link="https://repo.radeon.com/rocm/apt/5.7.1/pool/main/h/hip-runtime-amd/hip-runtime-amd_5.7.31921.50701-98~20.04_amd64.deb"
            download_and_extract_deb $download_link
        elif [ "$VERSION_ID" == "\"22.04\"" ]; then
            # Ubuntu 22.04
            download_link="https://repo.radeon.com/rocm/apt/5.7.1/pool/main/h/hip-runtime-amd/hip-runtime-amd_5.7.31921.50701-98~22.04_amd64.deb"
            download_and_extract_deb $download_link
        else
            # Unknown Ubuntu version
            echo "Unknown Ubuntu OS"
        fi
        ;;
    \"centos\")
        # CentOS
        if [ "$VERSION_ID" == "\"7\"" ]; then
            # CentOS 7
            download_link="https://repo.radeon.com/rocm/yum/5.7.1/main/hip-runtime-amd-5.7.31921.50701-98.el7.x86_64.rpm"
            download_and_extract_rpm $download_link
        else
            # Unknown CentOS version
            echo "Unknown CentOS"
        fi
        ;;
    \"rhel\")
        # RHEL
        if [ "$PLATFORM_ID" == "\"platform:el8\"" ]; then
            # RHEL 8
            download_link="https://repo.radeon.com/rocm/rhel8/5.7.1/main/hip-runtime-amd-5.7.31921.50701-98.el8.x86_64.rpm"
            download_and_extract_rpm $download_link
        elif [ "$PLATFORM_ID" == "\"platform:el9\"" ]; then
            # RHEL 9
            download_link="https://repo.radeon.com/rocm/rhel9/5.7.1/main/hip-runtime-amd-5.7.31921.50701-98.el9.x86_64.rpm"
            download_and_extract_rpm $download_link
        else
            # Unknown RHEL version
            echo "Unknown RHEL OS"
        fi
        ;;
    \"sles\")
        # SLES
        if [ "$VERSION_ID" == "\"15.4\"" ]; then
            # SLES 15SP4
            download_link="https://repo.radeon.com/rocm/zyp/5.7.1/main/hip-runtime-amd-5.7.31921.50701-sles154.98.x86_64.rpm"
            download_and_extract_rpm $download_link
        else
            # Unknown SLES version
            echo "Unknown SLES OS"
        fi
        ;;
    *)
        echo "Unknown OS"
        ;;
esac

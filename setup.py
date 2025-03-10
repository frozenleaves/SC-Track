#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: setup.py
# @Author: Li Chengxin 
# @Time: 2023/7/4 13:43

import setuptools

with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as fr:
    pkg_requirements = fr.read().split('\n')
    pkg_requirements.remove('')


VERSION = '0.0.10'

setuptools.setup(
    name='SC-Track',
    author="Li Chengxin",
    author_email="914814442@qq.com",
    url="https://github.com/chan-labsite/SC-Track",
    license="GNU General Public License v3.0",
    version=VERSION,
    description='SC-Track: A biologically inspired algorithm for accurate single cell tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=pkg_requirements,
    entry_points={
        'console_scripts': [
            'sctrack = SCTrack.sctrack:main',
        ],
    },
)

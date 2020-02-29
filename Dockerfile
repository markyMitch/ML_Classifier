FROM centos
RUN yum install python36 -y
RUN yum install git -y
RUN alias python='python3'
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install pandas
RUN python3 -m pip install xlrd
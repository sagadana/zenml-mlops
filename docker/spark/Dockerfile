FROM apache/spark:4.0.1

USER root

ARG HADOOP_VERSION=3.4.1
ARG AWS_SDK_V1_VERSION=1.12.720
ARG AWS_SDK_V2_BUNDLE_VERSION=2.24.6

RUN curl --fail --location --silent --show-error \
      --output /opt/spark/jars/hadoop-aws-${HADOOP_VERSION}.jar \
      https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar \
    && curl --fail --location --silent --show-error \
      --output /opt/spark/jars/aws-java-sdk-core-${AWS_SDK_V1_VERSION}.jar \
      https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-core/${AWS_SDK_V1_VERSION}/aws-java-sdk-core-${AWS_SDK_V1_VERSION}.jar \
    && curl --fail --location --silent --show-error \
      --output /opt/spark/jars/bundle-${AWS_SDK_V2_BUNDLE_VERSION}.jar \
      https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/${AWS_SDK_V2_BUNDLE_VERSION}/bundle-${AWS_SDK_V2_BUNDLE_VERSION}.jar

USER spark
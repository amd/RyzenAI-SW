// String branchName = "dev"
// String gitCredentials = ""
// String repoUrl = "https://github.com/username/repo-name.git"
pipeline {
    parameters {
        choice(name: 'PLATFORM_TYPE', choices: ['all', 'Ubuntu.20.04.x86_64', 'sdk-2022.2'], description: 'Run on specific platform')
        choice(name: 'BUILD_TYPE', choices: ['all', 'Release', 'Debug'], description: 'Run build type')
    }
    agent {
        node{
            label 'xcdl190091'
        }
    }
    environment {
        MY_SSH_KEY = credentials('zhanghui')
        GIT_SSH_COMMAND = "ssh -o StrictHostKeyChecking=no -i $MY_SSH_KEY"
        https_proxy = 'http://localhost:9181'

    }
    stages {
        stage('Generate passwd') {
            steps {
                sh "echo huizhang:x:30485:10585::/home/huizhang:/bin/sh > ${env.WORKSPACE}/passwd"
            }
        }
        stage('Test environment') {
            agent {
                docker {
                    image 'artifactory.xilinx.com/vitis-ai-docker-dev/aisw/dev:v3.0-158-gcd67e2f'
                    args "--network=host --entrypoint= -v ${env.WORKSPACE}/passwd:/etc/passwd -v ${env.WORKSPACE}:/home/huizhang"
                    reuseNode true
                }
            }
            steps {
                sh "cat $MY_SSH_KEY"
                sh("echo $MY_SSH_KEY")
                sh "rm -rf unilog || true"
                sh 'printenv; git clone git@gitenterprise.xilinx.com:VitisAI/unilog.git'
                sh "curl -kv https://www.google.com && echo ok"
            }
        }

        stage('Matrix Build') {
          matrix{
            agent {
                docker {
                    image 'artifactory.xilinx.com/vitis-ai-docker-dev/aisw/dev:v3.0-158-gcd67e2f'
                    args "--network=host --entrypoint= -v /group/modelzoo:/group/modelzoo -v ${env.WORKSPACE}/passwd:/etc/passwd -v ${env.WORKSPACE}:/home/huizhang"
                    reuseNode true
                }
            }
            axes {
                axis {
                    name 'PLATFORM'
                    values 'Ubuntu.20.04.x86_64', 'sdk-2022.2'
                }
                axis {
                    name 'TYPE'
                    values 'Release', 'Debug'
                }
            }
            when {allOf{
                    expression { return params.PLATFORM_TYPE == 'all' || params.PLATFORM_TYPE == PLATFORM }
                    expression { return params.BUILD_TYPE == 'all' || params.BUILD_TYPE == TYPE }

            }}
            stages {
              stage("build"){
                steps {
                  script{
                    if( PLATFORM == 'Ubuntu.20.04.x86_64' ){
                        sh "env CI_WORKSPACE=${env.WORKSPACE}/${PLATFORM}_${TYPE}_workspace python main.py --type ${TYPE}"
                    } else {
                        sh "unset LD_LIBRARY_PATH; . /opt/petalinux/2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux;env CI_WORKSPACE=${env.WORKSPACE}/${PLATFORM}_${TYPE}_workspace python main.py --type ${TYPE}"
                    }
                  }
                }
              }
            }
          }
        }
        stage("test"){
          agent {
              docker {
                  image 'artifactory.xilinx.com/vitis-ai-docker-dev/aisw/dev:v3.0-158-gcd67e2f'
                  args "--network=host --entrypoint= -v /group/modelzoo:/group/modelzoo -v ${env.WORKSPACE}/passwd:/etc/passwd -v ${env.WORKSPACE}:/home/huizhang"
                  reuseNode true
              }
          }
          steps {
            script{
              def PLATFORM = (params.PLATFORM_TYPE == 'Ubuntu.20.04.x86_64' || params.PLATFORM_TYPE == 'all') ? 'Ubuntu.20.04.x86_64' : params.PLATFORM_TYPE  
              def TYPE = (params.BUILD_TYPE == 'Debug' || params.BUILD_TYPE == 'all') ? 'Debug' : params.BUILD_TYPE
              if( PLATFORM == 'Ubuntu.20.04.x86_64' && TYPE == 'Debug' ){
                parallel (
                  1: { sh "env DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 XLNX_TARGET_NAME=DPUCZDX8G_ISA1_B4096 PATH=${env.WORKSPACE}/.local/${PLATFORM}.${TYPE}/bin:${env.PATH} LD_LIBRARY_PATH=${env.WORKSPACE}/.local/${PLATFORM}.${TYPE}/lib:/usr/lib/x86_64-linux-gnu test_onnx_runner /group/modelzoo/internal-cooperation-models/torchvision/resnet50/quantized/ResNet_int.onnx" },
                  2: { sh "env DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 XLNX_TARGET_NAME=DPUCZDX8G_ISA1_B4096 PATH=${env.WORKSPACE}/.local/${PLATFORM}.${TYPE}/bin:${env.PATH} LD_LIBRARY_PATH=${env.WORKSPACE}/.local/${PLATFORM}.${TYPE}/lib:/usr/lib/x86_64-linux-gnu test_onnx_runner /group/modelzoo/internal-cooperation-models/torchvision/inceptionv3/Inception3_int.onnx"}
                )
              }
            }
          }
        }
    }
}



#include <cerrno>
#include <ns3/core-module.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h> 
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UNSWRealisticSimulator");

// Global Sabitler
static std::string kTcpHost = "127.0.0.1";
static uint16_t kTcpPort = 5050;
static double kMeanInterArrival = 1.0;
static uint32_t kTotalFlows = 10000;

static std::string GenerateRandomIP(Ptr<UniformRandomVariable> uv)
{
    std::ostringstream ip;
    ip << static_cast<int>(uv->GetValue(1, 255)) << '.'
       << static_cast<int>(uv->GetValue(0, 256)) << '.'
       << static_cast<int>(uv->GetValue(0, 256)) << '.'
       << static_cast<int>(uv->GetValue(1, 255));
    return ip.str();
}

static uint32_t gFlowCount = 0;

static void GenerateUNSWRealisticLog()
{
    if (kTotalFlows && gFlowCount >= kTotalFlows) return;
    ++gFlowCount;
    static int sockfd = -1;
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();

    std::string srcIp = GenerateRandomIP(uv);
    std::string dstIp = GenerateRandomIP(uv);

    static int normalCounter = 0;
    const int normalsBeforeAttack = 10;
    static int attackSeq[9] = {0,1,2,3,4,5,6,7,8};
    static int attackIdx = 0;

    int patternType;
    if (normalCounter < normalsBeforeAttack) {
        patternType = 9;
        ++normalCounter;
    } else {
        patternType = attackSeq[attackIdx];
        attackIdx = (attackIdx + 1) % 9;
        normalCounter = 0;
    }

    // Ön tanımlı değişkenler
    double dur = uv->GetValue(0.10, 20.0);
    double sbytes = uv->GetValue(500, 50000);
    double dbytes = uv->GetValue(500, 50000);
    std::string state = "SF";
    std::string proto = "TCP";
    uint16_t trans_depth_local = uv->GetInteger(0, 5);
    uint32_t res_bdy_len_local = uv->GetInteger(0, 20000);
    uint8_t is_ftp_login = 0;
    uint8_t ct_ftp_cmd = 0;
    uint8_t ct_flw_http_mthd = 0;
    uint16_t ct_dst_ltm = 1;
    uint16_t ct_dst_sport_ltm = 0;
    uint16_t ct_dst_src_ltm = 1;

    std::string attackLabel = "Normal";
    switch (patternType) {
    case 0:
        attackLabel = "DoS";
        sbytes = uv->GetValue(45000, 60000);
        dur = uv->GetValue(0.1, 1.5);
        state = "REJ";
        break;
    case 1:
        attackLabel = "Reconnaissance";
        ct_dst_ltm = 9;
        ct_dst_sport_ltm = 5;
        dur = uv->GetValue(0.01, 0.15);
        break;
    case 2:
        attackLabel = "Fuzzers";
        sbytes = uv->GetValue(25000, 50000);
        break;
    case 3:
        attackLabel = "Exploits";
        trans_depth_local = uv->GetInteger(5, 10);
        dbytes = uv->GetValue(35000, 50000);
        break;
    case 4:
        attackLabel = "Backdoor";
        is_ftp_login = 1;
        ct_ftp_cmd = 2;
        break;
    case 5:
        attackLabel = "Analysis";
        ct_flw_http_mthd = 1;
        res_bdy_len_local = uv->GetInteger(6000, 12000);
        break;
    case 6:
        attackLabel = "Generic";
        proto = "ICMP";
        dur = uv->GetValue(0.1, 0.5);
        break;
    case 7:
        attackLabel = "Shellcode";
        res_bdy_len_local = uv->GetInteger(8000, 15000);
        break;
    case 8:
        attackLabel = "Worms";
        ct_dst_src_ltm = 10;
        break;
    case 9:
    default:
        attackLabel = "Normal";
        dur = uv->GetValue(0.5, 2.0);
        sbytes = uv->GetValue(50, 200);
        dbytes = uv->GetValue(50, 200);
        break;
    }

    uint16_t sport = uv->GetInteger(1024, 65535);
    std::vector<uint16_t> commonPorts = {80, 443, 22, 21, 25, 8080};
    uint16_t dport = commonPorts[uv->GetInteger(0, commonPorts.size()-1)];
    uint16_t swin = uv->GetInteger(0, 255);
    uint16_t spkts = uv->GetInteger(2, 300);
    uint16_t dpkts = uv->GetInteger(2, 300);

    double sinpkt_local = dur / std::max(1.0, double(spkts));
    double dinpkt_local = dur / std::max(1.0, double(dpkts));
    double sjit_local = uv->GetValue(0, 1);
    double djit_local = uv->GetValue(0, 1);
    uint16_t ct_srv_src_local = uv->GetInteger(1, 10);
    uint16_t ct_srv_dst_local = uv->GetInteger(1, 10);

    uint8_t is_sm_ips_ports = uv->GetInteger(0, 1);
    uint16_t sttl = uv->GetInteger(32, 255);
    uint16_t dttl = uv->GetInteger(32, 255);
    double smean = sbytes / std::max(spkts, (uint16_t)1);
    double dmean = dbytes / std::max(dpkts, (uint16_t)1);
    double sloss = uv->GetInteger(0, 10);
    double dloss = uv->GetInteger(0, 10);
    double rate = uv->GetValue(0.1, 10.0);
    uint16_t synack = uv->GetInteger(0, 500);
    uint16_t ackdat = uv->GetInteger(0, 500);
    uint32_t ct_src_ltm = uv->GetInteger(1, 50);
    uint32_t ct_src_dport_ltm = uv->GetInteger(1, 50);
    uint32_t stcpb = uv->GetInteger(0, 1e7);
    uint32_t dtcpb = uv->GetInteger(0, 1e7);
    uint32_t swinn = uv->GetInteger(0, 255);
    uint32_t dwin = uv->GetInteger(0, 255);
    uint32_t response_body_len = uv->GetInteger(0, 20000);
    double tcprtt = uv->GetValue(0.001, 3.0);
    double sload = sbytes / dur;
    double dload = dbytes / dur;
    uint32_t id = uv->GetInteger(1, 65535);

    std::ostringstream js;
    js << std::fixed << std::setprecision(2)
       << "{"
       << "\"srcip\":\"" << srcIp << "\"," 
       << "\"dstip\":\"" << dstIp << "\"," 
       << "\"proto\":\"" << proto << "\"," 
       << "\"service\":\"-\"," 
       << "\"sport\":" << sport << ","
       << "\"state\":\"" << state << "\"," 
       << "\"dur\":" << dur << "," 
       << "\"sbytes\":" << sbytes << "," 
       << "\"spkts\":" << spkts << "," 
       << "\"dpkts\":" << dpkts << "," 
       << "\"swin\":" << swin << "," 
       << "\"dbytes\":" << dbytes << "," 
       << "\"sinpkt\":" << sinpkt_local << "," 
       << "\"dinpkt\":" << dinpkt_local << "," 
       << "\"sjit\":" << sjit_local << "," 
       << "\"djit\":" << djit_local << "," 
       << "\"trans_depth\":" << trans_depth_local << "," 
       << "\"res_bdy_len\":" << res_bdy_len_local << "," 
       << "\"ct_state_ttl\":0," 
       << "\"ct_flw_http_mthd\":0," 
       << "\"is_ftp_login\":0," 
       << "\"ct_ftp_cmd\":0," 
       << "\"ct_srv_src\":" << ct_srv_src_local << "," 
       << "\"ct_srv_dst\":" << ct_srv_dst_local << "," 
       << "\"ct_dst_ltm\":1," 
       << "\"ct_dst_sport_ltm\":0," 
       << "\"ct_dst_src_ltm\":1," 
       << "\"is_sm_ips_ports\":" << static_cast<int>(is_sm_ips_ports) << "," 
       << "\"sttl\":" << sttl << "," 
       << "\"dttl\":" << dttl << "," 
       << "\"smean\":" << smean << "," 
       << "\"dmean\":" << dmean << "," 
       << "\"sloss\":" << sloss << "," 
       << "\"dloss\":" << dloss << "," 
       << "\"rate\":" << rate << "," 
       << "\"synack\":" << synack << "," 
       << "\"ackdat\":" << ackdat << "," 
       << "\"ct_src_ltm\":" << ct_src_ltm << "," 
       << "\"ct_src_dport_ltm\":" << ct_src_dport_ltm << "," 
       << "\"stcpb\":" << stcpb << "," 
       << "\"dtcpb\":" << dtcpb << "," 
       << "\"swinn\":" << swinn << "," 
       << "\"dwin\":" << dwin << "," 
       << "\"response_body_len\":" << response_body_len << "," 
       << "\"tcprtt\":" << tcprtt << "," 
       << "\"sload\":" << sload << "," 
       << "\"dload\":" << dload << "," 
       << "\"id\":" << id
       << "}\n";

    // Yollanacak veri
if (sockfd == -1) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in serv{};
        serv.sin_family = AF_INET;
        serv.sin_port = htons(kTcpPort);
        inet_pton(AF_INET, kTcpHost.c_str(), &serv.sin_addr);
        if (connect(sockfd, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) != 0) {
            std::cerr << "[CONNECT-ERR] " << strerror(errno) << std::endl;
            close(sockfd);
            sockfd = -1;
        }
    }

    //  Veri gönderimi
    const std::string& payload = js.str();
    if (sockfd != -1) {
        ssize_t sent = send(sockfd, payload.c_str(), payload.size(), MSG_NOSIGNAL);
        std::cout << "[SENT] " << sent << " bytes\n";
        if (sent < 0) {
            std::cerr << "[SEND-ERR] " << strerror(errno) << std::endl;
            close(sockfd);
            sockfd = -1;  // yeniden bağlanmak için
        }
    }

    //  Üstel dağılımla zamanlama
    Ptr<ExponentialRandomVariable> expGap = CreateObject<ExponentialRandomVariable>();
    expGap->SetAttribute("Mean", DoubleValue(kMeanInterArrival));
    Simulator::Schedule(Seconds(expGap->GetValue()), &GenerateUNSWRealisticLog);
}

int main(int argc, char* argv[])
{
    CommandLine cmd;
    cmd.AddValue("rate",  "Mean inter-arrival (s)", kMeanInterArrival);
    cmd.AddValue("flows", "Total flows (0=infinite)", kTotalFlows);
    cmd.AddValue("host",  "TCP target IP", kTcpHost);
    cmd.AddValue("port",  "TCP target port", kTcpPort);
    cmd.Parse (argc, argv);

    Simulator::ScheduleNow (&GenerateUNSWRealisticLog);
    Simulator::Run ();
    Simulator::Destroy ();
    return 0;
}

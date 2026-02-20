// ============================================================
// NetworkSlicing.ned
// Topologie simu5G pour simulation de Network Slicing 5G
// 3 slices : eMBB, URLLC, mMTC + Best Effort
//
// Architecture correcte simu5G (3GPP 38.801 Option 2) :
//   UE <--radio--> gNodeB <--> iUpf <--> upf <--> router <--> Serveurs
//
// Chaque slice a ses propres UEs et son propre serveur applicatif.
// Le gNodeB applique un scheduler differenicie par slice.
// ============================================================

// No package - standalone simulation file

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.world.radio.LteChannelControl;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.common.binder.Binder;
import simu5g.nodes.gNodeB;
import simu5g.nodes.NrUe;
import simu5g.nodes.Upf;

network NetworkSlicing
{
    parameters:
        int numEmbbUes = default(4);   // UEs video cloud gaming
        int numUrllcUes = default(4);  // UEs controle temps-reel
        int numMmtcUes = default(2);   // UEs signalisation / admin
        int numBestEffortUes = default(2);

        @display("bgb=1000,600");

    submodules:
        // --- Infrastructure simu5G ---
        channelControl: LteChannelControl {
            @display("p=60,25;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=180,25;is=s");
        }
        binder: Binder {
            @display("p=300,25;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=420,25;is=s");
        }

        // --- Coeur reseau 5G (architecture 3GPP) ---
        // gNodeB -> iUpf -> upf -> router -> serveurs
        gnb: gNodeB {
            @display("p=150,300;is=vl");
        }
        iUpf: Upf {
            @display("p=350,300;is=l");
        }
        upf: Upf {
            @display("p=500,300;is=l");
        }
        router: Router {
            @display("p=650,300;is=l");
        }

        // --- Serveurs applicatifs (un par slice) ---
        serverEmbb: StandardHost {
            @display("p=850,150;is=n");
        }
        serverUrllc: StandardHost {
            @display("p=850,250;is=n");
        }
        serverMmtc: StandardHost {
            @display("p=850,350;is=n");
        }
        serverBestEffort: StandardHost {
            @display("p=850,450;is=n");
        }

        // --- UEs par slice ---
        ueEmbb[numEmbbUes]: NrUe {
            @display("p=50,150");
        }
        ueUrllc[numUrllcUes]: NrUe {
            @display("p=50,250");
        }
        ueMmtc[numMmtcUes]: NrUe {
            @display("p=50,350");
        }
        ueBestEffort[numBestEffortUes]: NrUe {
            @display("p=50,450");
        }

    connections allowunconnected:
        // === Architecture 5G Core (identique a SingleCell_Standalone.ned) ===

        // gNodeB <-> iUpf (intermediate UPF - N3 interface)
        gnb.ppp <--> Eth10G <--> iUpf.pppg++;

        // iUpf <-> upf (N9 interface entre UPFs)
        iUpf.pppg++ <--> Eth10G <--> upf.pppg++;

        // upf <-> router (N6 interface - filterGate pour routage IP)
        upf.filterGate <--> Eth10G <--> router.pppg++;

        // === Data Network : Router <-> Serveurs applicatifs ===
        router.pppg++ <--> Eth10G <--> serverEmbb.pppg++;
        router.pppg++ <--> Eth10G <--> serverUrllc.pppg++;
        router.pppg++ <--> Eth10G <--> serverMmtc.pppg++;
        router.pppg++ <--> Eth10G <--> serverBestEffort.pppg++;
}

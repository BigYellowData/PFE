// ============================================================
// NetworkSlicing.ned
// Topologie simu5G pour simulation de Network Slicing 5G
// 3 slices : eMBB, URLLC, mMTC + Best Effort
//
// Architecture :
//   UE (mobile) <--5G NR--> gNodeB <--backhaul--> UPF <--N6--> Serveur
//
// Chaque slice a ses propres UEs et son propre serveur applicatif.
// Le gNodeB applique un scheduler differenicie par slice.
// ============================================================

// No package - standalone simulation file

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.world.radio.LteChannelControl;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.common.binder.Binder;
import simu5g.nodes.gNodeB;
import simu5g.nodes.NrUe;

network NetworkSlicing
{
    parameters:
        int numEmbbUes = default(4);   // UEs video cloud gaming
        int numUrllcUes = default(4);  // UEs controle temps-reel
        int numMmtcUes = default(2);   // UEs signalisation / admin
        int numBestEffortUes = default(2);

        @display("bgb=900,600");

    submodules:
        // --- Infrastructure ---
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

        // --- Coeur reseau (routeur simple au lieu de UPF simu5G) ---
        upf: Router {
            @display("p=450,300;is=l");
        }

        // --- gNodeB (station de base 5G NR) ---
        gnb: gNodeB {
            @display("p=250,300;is=vl");
        }

        // --- Serveurs applicatifs (un par slice) ---
        serverEmbb: StandardHost {
            @display("p=650,150;is=n");
        }
        serverUrllc: StandardHost {
            @display("p=650,250;is=n");
        }
        serverMmtc: StandardHost {
            @display("p=650,350;is=n");
        }
        serverBestEffort: StandardHost {
            @display("p=650,450;is=n");
        }

        // --- UEs par slice ---
        ueEmbb[numEmbbUes]: NrUe {
            @display("p=100,150");
        }
        ueUrllc[numUrllcUes]: NrUe {
            @display("p=100,250");
        }
        ueMmtc[numMmtcUes]: NrUe {
            @display("p=100,350");
        }
        ueBestEffort[numBestEffortUes]: NrUe {
            @display("p=100,450");
        }

    connections allowunconnected:
        // Backhaul : gNodeB <-> UPF (Router)
        gnb.ppp <--> { datarate = 10Gbps; delay = 1ms; } <--> upf.pppg++;

        // N6 : UPF <-> Serveurs applicatifs
        upf.pppg++ <--> { datarate = 10Gbps; delay = 5ms; } <--> serverEmbb.pppg++;
        upf.pppg++ <--> { datarate = 10Gbps; delay = 1ms; } <--> serverUrllc.pppg++;
        upf.pppg++ <--> { datarate = 1Gbps;  delay = 10ms; } <--> serverMmtc.pppg++;
        upf.pppg++ <--> { datarate = 1Gbps;  delay = 5ms; } <--> serverBestEffort.pppg++;
}
